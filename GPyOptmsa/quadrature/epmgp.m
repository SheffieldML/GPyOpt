%%%%%%%%%%%%%%%%%%%%%%%
% John P Cunningham
% 2011
%
% Uses EP to calculate a multivariate normal probability
% and the moments of the corresponding truncated multivariate Gaussian.
% 
% This is a general version of the EPGCD algorithm from Cunningham 2009
% Thesis, generalized to calculate over arbitrary intersection of
% halfspaces, aka possibly unbounded convex polyhedra.
%
% Uses a numerically stable implementation of both the truncated univariate
% Gaussian moments and the EP steps themselves.  
%
% Inputs: 
% --parameters of the Gaussian--
%  m, the mean of the multivariate Gaussian
%  K, the positive semi-definite covariance matrix of the Gaussian
% --parameters of the integration region--
%  C, the matrix of columns that define the halfspaces via that
%      uB(i) > c(:,i)'*x > lB(i).  Note that the columns should be unit norm, so we check for
%      that and warn and correct.  Note that this matrix is dimension n by p,
%      where n is the length of m, and n is arbitrary.
%  lB, the lower bounds lB  such that c(:,i)'*x > lB(i). 
%  uB the same, but upper bound.
%
% Outputs:
%  logZEP, the log probability calculated by EPMGP
%  mu, the mean of the truncated Gaussian (calculated by EPMGP)
%  Sigma, the covariance of the truncated Gaussian (calculated by EPMGP)
%  extras, other potentially interesting summary statistics.
%
% Note: These results are approximations, but approximations with high
% accuracy.  See Cunningham, Hennig, Lacoste-Julien (2011) Manuscript for the evaluation
% of the accuracy.
%%%%%%%%%%%%%%%%%%%%%%%
function [logZ, mu, Sigma, extras] = epmgp(m,K,C,lB,uB,maxSteps,alphaCorrection)

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % some useful parameters
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    n = length(m);
    p = size(C,2);
    makeExtras = 0; % for saving extra information, debugging etc.
    errorCheck = 1; % a good idea, but it costs a bit more computation...
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % first some error checking
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if nargin < 7 || isempty(alphaCorrection)
        alphaCorrection = 1;
    end
    if nargin < 6 || isempty(maxSteps)
        maxSteps = 200;
    end
    if errorCheck
        % check sizes
        if n~=size(C,1) || n~=size(K,1) || n~=size(K,2)
            fprintf('ERROR: the mean vector is not the same size as the columns of C or K (or K is not square).\n');
            keyboard
        end
        % check norms of C.
        if any(abs(sum(C.*C,1)-1)>1e-6)
            fprintf('WARNING: the columns of C should be unit norm.  We will correct that here, but you better make sure C, lB, uB are correct.\n');
            %fprintf('C^T*W should change, which will change the answer, but we will correct that... Check C, W, Cold, Wold before continuing...\n');
            Cold = C;
            %Wold = W;
            C = C./repmat(sqrt(sum(C.*C)),n,1);
            %W = W.*repmat(sqrt(sum(C.*C)),n,1);
            %keyboard
        end
        % check Sigma
        if norm(K - K')>1e-14;
            fprintf('ERROR: your covariance matrix is not symmetric.\n');
            keyboard;
        end
        % check ub and lb
        if size(lB,1)~=p
            %
            fprintf(2,'ERROR: lB should be a p vector.\n');
        end
        %
        if nargin < 5 || isempty(uB)
            % no upper bound given... assume inf
            uB = inf(size(lB));
        end
        %
        % Further, if K is diagonal or somehow otherwise uninteresting, the
        % answer might be similarly constrained and/or uninteresting.  Careful
        % about this.
        if (norm(K-diag(diag(K)))<1e-14)
            % diagonal K
            fprintf('WARNING: your initial Sigma is diagonal... you better know what you are doing. Continuing...\n');
        end
        
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % convergence criteria for stopping EPMGP
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    epsConverge = 1e-8; % algorithm is not particularly sensitive to this choice
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % initialize algorithm parameters
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    nuSite = zeros(p,1);
    tauSite = zeros(p,1);
    % the following should never be used, so we just prealovate them as
    % placeholders (and I suppose a bit of performance).
    nuCavity = nan(p,1);
    tauCavity = nan(p,1);
    deltaTauSite = nan(p,1);
    deltaNuSite = nan(p,1);
    logZhat = nan(p,1);
    muhat = nan(p,1);
    sighat = nan(p,1);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % make the power EP heuristic...
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % the frac terms appear in three places: calc of cavity, calc of sites
    % from fractional sites, and in the final logZ calc
    fracTerms = 1*(1./sum(abs(C'*C),1))';
    fracTerms = 1*alphaCorrection*ones(size(fracTerms));
    dampTerms = 1*ones(p,1);
    dampRedamp = 0.8;
    
    if makeExtras
        extras.m = m;
        extras.K = K;
        extras.C = C;
        extras.lB = lB;
        extras.uB = uB;
        extras.fracTerms = fracTerms;
        extras.dampTerms = dampTerms;
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % initialize the distribution q(x)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % algorithm is not sensitive to this choice.  Testing suggests a unimodal
    % optimization surface.  This is a reasonable q(x) initial point.
    Sigma = K;
    mu = m;
    % Note that the initialization point may require further investigation
    % later.  If the algorithm is unstable, it might be sensible to use a
    % starting point near the eventual answer, such as some center of the
    % region A.  This could be an LP and requires an LP solver, which is fine but
    % may be overkill here.  Another approach is to generate a bunch of
    % halfspace boundary points and to use their centroid, using centroid()
    % (see
    % http://www.mathworks.com/matlabcentral/fileexchange/8514-centroid-of-a-
    % convex-n-dimensional-polyhedron )
    % and the function convhulln, which would give something akin to the axis
    % aligned initilization of: mu = mean([lowerB upperB],2);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % precalculate and initialize a few quantities
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    L = chol(K,'lower');
    
    logZ = Inf;
    muLast = -Inf*ones(size(mu));
    converged = 0;
    k = 1;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % MAIN EPMGP ALGORITHM LOOP
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    while(~converged && k<maxSteps)
        
        % if we have started a new iteration, we never want to
        % automatically restart the algorithm.
        restartAlgorithm = 0;
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % INNER ALGORITHM LOOP
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %{
        if mod(k,2)==0
            indprogress = [1:p];
        else
            indprogress = [p:-1:1];
        end
        %}
        for j = 1:p 
            % we must iterate over each site.
            skipSite = 0;
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % make the cavity distribution
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            tauCavity(j) = 1/(C(:,j)'*Sigma*C(:,j)) - tauSite(j)/fracTerms(j);
            nuCavity(j) = (C(:,j)'*mu)./(C(:,j)'*Sigma*C(:,j)) - nuSite(j)/fracTerms(j);
            
            
            if tauCavity(j) <= 0 
                % problem... negative cavity updates...
                fprintf(2,'ERROR: We have a negative tauCavity... Investigate (p = %d , k = %d , j = %d).\n',p,k,j); 
                fprintf(2,'This typically happens with too much redundancy in constraints, and the sites go into an unstable oscillation.\n')
                fprintf(2,'We will restart the algorithm with more damping (from %0.2f to %0.2f) to (hopefully) fix this.\n',dampTerms(j), dampRedamp*dampTerms(j));
                %tauCavity(j) = max(1e-4, tauCavity(j));
                restartAlgorithm = 1;
                dampTerms(j) = dampTerms(j)*dampRedamp;
                %fracTerms(j) = fracTerms(j)/dampRedamp;
                %keyboard
                break;
                % this should restart things.
                % just skip this site
                % does not work. skipSite = 1;
            end
            
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % compute moments using truncated normals
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            if uB(j) < lB(j)
                % error
                fprintf(2,'ERROR: upperB smaller than lowerB?!?\n');
                keyboard
            end
            % A very numerically stable implementation is avaible in
            % truncNormMoments, which we use here:
            [logZhat(j), ~, muhat(j), sighat(j)] = truncNormMoments(lB(j),uB(j),nuCavity(j)/tauCavity(j),1/tauCavity(j));
     
            if sighat(j)==0
                % then the algorithm has found a zero weight dimension, and
                % the algorithm should terminate.
                fprintf('0 moment found. exiting...');
                converged = 1;
                logZ = -Inf;
                mu = NaN;
                Sigma = NaN;
                break;
            end
            
    
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % update the site parameters
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            deltaTauSite(j) = dampTerms(j)*(fracTerms(j)*(1/sighat(j) - tauCavity(j)) - tauSite(j));
            deltaNuSite(j) =  dampTerms(j)*(fracTerms(j)*(muhat(j)/sighat(j) - nuCavity(j)) - nuSite(j));
            tauSite(j) = tauSite(j) + deltaTauSite(j);
            nuSite(j) = nuSite(j) + deltaNuSite(j);
            
            if tauSite(j)<0
                % tauSite is provably nonnegative.  If a result is negative,
                % it is either numerical precision (in which case, we correct that
                % to 0) or an error in the algorithm.
                if tauSite(j,:) > -1e-6
                    tauSite(j,:) = 0;
                else
                    fprintf('ERROR.  Negative tauSite can not happen. Please check code.\n');
                    keyboard
                end
            end
            
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            % Now update q(x) (Sigma and mu)
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            sc = Sigma*C(:,j);
            Sigma = Sigma - (deltaTauSite(j)/(1 + deltaTauSite(j)*(C(:,j)'*sc)))*(sc*sc');
            mu = mu + ((deltaNuSite(j) - deltaTauSite(j)*(C(:,j)'*mu))/(1 + deltaTauSite(j)*(C(:,j)'*sc)))*sc;
            
        end
        % we never check the convergence criteria mid loop, as there may be
        % an uninformative site that doesn't change anything, so we don't
        % want to preemptively exit.
        %{
             % debug check... most basic formulation
           Si = inv(K);
           for i = 1 : length(tauSite)
               % make a new Sigma
               Si = Si + tauSite(i)*C(:,i)*C(:,i)';
           end
           Sigma = inv(Si);
           
           mu = Sigma*(K\m + sum(C.*repmat(nuSite',n,1),2));
          %}  
       
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % check convergence criteria
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if ((norm(muLast-mu)) < epsConverge)
            converged=1;
        else
            muLast = mu;
        end
        
        if makeExtras
            extras.iter(k).tauSite = tauSite;
            extras.iter(k).nuSite = nuSite;
            extras.iter(k).mu = mu;
            extras.iter(k).Sigma = Sigma;
            extras.iter(k).tauCavity = tauCavity;
            extras.iter(k).nuCavity = nuCavity;
            extras.iter(k).logZhat = logZhat;
            extras.iter(k).muhat = muhat;
            extras.iter(k).sighat = sighat;
            extras.FailedAttempts = [];
        end
        
        k = k+1;
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % if sites are oscillating, restart everything
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        if restartAlgorithm
            k = 1;
            converged = 0;
            % increase the damping
            dampTerms = dampRedamp*dampTerms;
            % reinitialize
            nuSite = zeros(p,1);
            tauSite = zeros(p,1);
            mu = m;
            Sigma = K;
            if makeExtras
                extras.FailedAttempts = extras;
                extras = rmfield(extras,'iter');
                extras.fracTerms = fracTerms;
                extras.dampTerms = dampTerms;
            end
            
        end
    
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % FINISHED ALGORITHM MAIN LOOP
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %              fprintf('Progress check (p = %d , k = %d , j = %d).\n',p,k,j);
    %keyboard

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % compute logZ, the EP MGP result, from q(x)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if logZ ~= -Inf % ensure that no 0 moment was found, in which case this calculation is not needed
        muCavity = nuCavity./tauCavity;
        sigCavity = 1./tauCavity;
        % now this must be looped
        lZdetmat = eye(n);
        for j = 1:p
            lc = L'*C(:,j);
            % add the next site outer product
            lZdetmat = lZdetmat + tauSite(j)*lc*lc';
        end
        lZ0 = -0.5*logdet(lZdetmat);
        lZ1 = -0.5*norm(L\m)^2 + 0.5*norm(L\mu)^2 + 0.5*sum(tauSite.*((C'*mu).^2)) ;
        % before frac terms...
        %lZ2 = sum(logZhat) + 0.5*sum(log1p(tauSite.*sigCavity));
        %lZ3 = 0.5*sum( (muCavity.^2.*tauSite - 2*muCavity.*nuSite - nuSite.^2.*sigCavity) ./ (1 + tauSite.*sigCavity) );
        lZ2 = sum(fracTerms.*logZhat) + 0.5*sum(fracTerms.*log1p(tauSite.*sigCavity./fracTerms));
        lZ3 = 0.5*sum( (muCavity.^2.*tauSite.*fracTerms - 2*muCavity.*nuSite.*fracTerms - nuSite.^2.*sigCavity) ./ ( fracTerms + tauSite.*sigCavity) );
        logZ = lZ0 + lZ1 + lZ2 + lZ3;
    end
    
    % here are some sanity checks that hopefully will not be necessary
    % again...
    %{
    % lZ0 ok...
    lZcheck0 = -0.5*logdet(K) + 0.5*logdet(Sigma); 
    % lZ1 ok...
    lZcheck1 = -0.5*m'*(K\m) + 0.5*mu'*(Sigma\mu);
    % these are still wrong... but that's these equations, not our desired
    % calcs above...
    lZcheck2 = 0.5*sum(log(tauSite)) - 0.5*sum((nuSite.^2).*tauSite) + sum(logZhat)  + 0.5*sum(log(sigCavity + 1./tauSite));
    lZcheck3 = 0.5*sum(((muCavity - nuSite./tauSite).^2)./(sigCavity + 1./tauSite)) ;
    lZcheck = lZcheck0 + lZcheck1 + lZcheck2 + lZcheck3;
    %
    keyboard
    %}
    
    if makeExtras
        extras.logZ = logZ;
        extras.mu = mu;
        extras.Sigma = Sigma;
    else
        extras = [];
    end
    
    
    
    
end


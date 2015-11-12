%%%%%%%%%%%%%%%%%%%%%%%
% John P Cunningham
% 2008
% updated 2011 for numerical stability, so it extends to a much larger
% range of tail probabilities now.
%
% Uses EP to calculate a multivariate gaussian probability
% and the moments of the corresponding truncated multivariate Gaussian.
%
% Uses the numerically stable EP implementation from
% Rasmussen and Williams (2006), "Gaussian Processes for Machine Learning"
% section 3.6 (p 58), adapted to the EPMGP problem specifics.  Only the
% part below "Compute moments using truncated normals" is different from
% the implementation of Rasmussen and Williams. 
%
% Inputs: 
% --parameters of the Gaussian--
%  m, the mean of the multivariate Gaussian
%  K, the positive semi-definite covariance matrix of the Gaussian
% --parameters of the integration region--
%  lowerB, the lower bound of the hyper-rectangle (a vector of the same size as m)
%  upperB, the upper bound of the hyper-rectangle (a vector of the same size as m)
%
% Outputs:
%  logZEP, the log cumulative density calculated by EPMGP
%  mu, the mean of the truncated Gaussian (calculated by EPMGP)
%  Sigma, the covariance of the truncated Gaussian (calculated by EPMGP)
%%%%%%%%%%%%%%%%%%%%%%%
function [logZEP, mu, Sigma] = axisepmgp(m,K,lowerB,upperB)

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % convergence criteria for stopping EPGCD
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  epsConverge = 1e-5; % algorithm is not particularly sensitive to this choice

  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % initialize algorithm parameters
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  nuSite = zeros(size(K,1),1);
  tauSite = zeros(size(K,1),1);
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % initialize the distribution q(x)
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % algorithm is not sensitive to this choice.  Testing suggests a unimodal
  % optimization surface.  This is a reasonable q(x) initial point.
  Sigma = K; 
  mu = mean([lowerB upperB],2); 
  for i = 1:length(mu)
      if isinf(mu(i))
          % just to make sure that mu doesn't get initialized to an
          % unacceptable number.
          mu(i) = sign(mu(i))*100;
      end
  end
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % precalculate and initialize a few quantities
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  KinvM = K\m;
  logZEP = Inf;
  muLast = -Inf*ones(size(mu));
  converged = 0;
  k = 1;
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % MAIN axisEPMGP ALGORITHM LOOP
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  while(~converged)

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % make the cavity distribution
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    tauCavity = 1./diag(Sigma) - tauSite;
    nuCavity = mu./diag(Sigma) - nuSite;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % compute moments using truncated normals
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % A numerically stable implementation is avaible in
    % truncNormMoments, which we use here:
    [logZhat, ~, muhat, sighat] = truncNormMoments(lowerB,upperB,nuCavity./tauCavity,1./tauCavity);    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % update the site parameters
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    deltatauSite = 1./sighat - tauCavity - tauSite;
    tauSite = tauSite + deltatauSite;
    nuSite = muhat./sighat - nuCavity;
    % tauSite is provably nonnegative.  If a result is negative,
    % it is either numerical precision (in which case, we correct that
    % to 0) or an error in the algorithm.
    if any(tauSite)<0
      for i=1:length(tauSite)
        if tauSite(i,:) > -1e-8
          tauSite(i,:) = 0;
        else
          fprintf('ERROR.  This can not happen. Please check code.\n');
          keyboard
        end
      end  
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % having now iterated through all sites, update q(x) (Sigma and mu)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    SsiteHalf = diag(sqrt(tauSite));
    L = chol(eye(size(K)) + SsiteHalf*K*SsiteHalf);
    V = L'\(SsiteHalf*K);
    Sigma = K - V'*V;
    mu = Sigma*(nuSite + KinvM);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % check convergence criteria
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if ((norm(muLast-mu)) < epsConverge)
      converged=1;
    else
      muLast = mu;
    end
    k = k+1;

  end
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % FINISHED ALGORITHM MAIN LOOP
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  % compute logZEP, the EPMGP probability, from q(x)
  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
  if logZEP ~= -Inf % ensure that no 0 moment was found, in which case this calculation is not needed
    lZ1 = 0.5*sum(log( 1 + tauSite./tauCavity)) - sum(log(diag(L)));
    lZ2 = 0.5*(nuSite - tauSite.*m)'*(Sigma - diag(1./(tauCavity + tauSite)))*(nuSite - tauSite.*m); 
    lZ3 = 0.5*nuCavity'*((diag(tauSite) + diag(tauCavity))\(tauSite.*nuCavity./tauCavity - 2*nuSite));
    lZ4 = - 0.5*(tauCavity.*m)'*((diag(tauSite) + diag(tauCavity))\(tauSite.*m - 2*nuSite));  
    logZEP = lZ1 + lZ2 + lZ3 + lZ4 + sum(logZhat);
  end
 
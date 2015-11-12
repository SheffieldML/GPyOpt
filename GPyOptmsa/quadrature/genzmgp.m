%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% John P Cunningham
% 2011
%
% genzmgp.m
%
% This is a wrapper function for the software provided by Alan Genz for
% lattice point MGP calculation, available at the time of writing as
% qsclatmvnv.m at
% http://www.math.wsu.edu/faculty/genz/software/software.html
%
% This function just formats the inputs and outputs as expected by testMGP
% and other methods.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [ logmgp , mgp , errGenz ] = genzmgp( m , K , C , lB , uB , numPoints)

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % some useful parameters
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    errorCheck = 0; % cleaner, but requires a bit more computation...
    n = length(m);
    p = size(C,2);
    % number of integration points to use with Genz method
    if nargin < 6 || isempty(numPoints)
        % then set it to a big number
        numPoints = 50000;
    end
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % first some error checking
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if errorCheck
        % check sizes
        if n~=size(C,1) || n~=size(K,1) || n~=size(K,2)
            fprintf('ERROR: the mean vector is not the same size as the columns of C or K (or K is not square).\n');
            keyboard
        end
        % check norms of C.
        if any(abs(sum(C.*C,1)-1)>1e-6)
            fprintf('WARNING: the columns of C should be unit norm.  We will correct that here, but you better make sure C and W are correct.\n');
            %fprintf('C^T*W should change, which will change the answer, but we will correct that... Check C, W, Cold, Wold before continuing...\n');
            Cold = C;
            %Wold = W;
            C = C./repmat(sqrt(sum(C.*C)),n,1);
            %W = W.*repmat(sqrt(sum(C.*C)),n,1);
        end
        % check Sigma
        if norm(K - K')>1e-14;
            fprintf('ERROR: your covariance matrix is not symmetric.\n');
            keyboard;
        end
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % now format as expected by QSCLATMVNV
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    a = lB - C'*m; % must adjust the bounding box to be around a zero centered distribution
    b = uB - C'*m; % by definition the halfspace always extends to infty

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % call and return
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    [ mgp , errGenz ] = qsclatmvnv( numPoints, K , a , C' , b );
    % return
    logmgp = log(mgp);
    
end

    
    
    
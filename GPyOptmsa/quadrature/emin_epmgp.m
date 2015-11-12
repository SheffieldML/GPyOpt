%%%%%%%%%%%%%%%%%%%%%%%
% Javier Gonzalez
% 2015
%
% This function uses EP to calculate the expectation of min(y,eta) where 
% y is a Gaussian  multivariate vector with mean m and covariance K and 
% eta is a real number.
% 
% This code is based is based on the EPMGP algorithm of Cunningham 2009 
% et al., 2011 that allows to compute Gaussian probabilities over arbitrary 
% intersection of (possibly unbounded) convex polyhedra.
%
% The idea of the metod is to split the integral over the min(y,eta) into
% a set of polyhedra and use the EPMGP method in each of those. 
%
% Inputs: 
%  m, the mean of the multivariate Gaussian
%  K, the positive semi-definite covariance matrix of the Gaussian
%  eta, upper bound for minimum of y
%
% Outputs:
%  emin, value of the integral E[min(y,eta)] with respect to p(y)~N(m,K)
%
%%%%%%%%%%%%%%%%%%%%%%



function [e_min,Int_y,Probs_y,Int_eta] = emin_epmgp(m,K,eta)

    % dimension of the problem
    n  = length(m);
    Int_y = zeros(1,n); 
    Probs_y = zeros(1,n);
    
    % --------------------------------------------------------
    % ---- Compute the integrals asociated to the first term
    % --------------------------------------------------------
    lB = [-Inf;zeros(n-1,1)];
    uB = [eta;Inf(n-1,1)];  % bounds are always the same for all j
    
    for j=1:n
       % Commpute C_j  
       Ij        = zeros(n,1);
       Ij(j)     = 1;
       Iaux      = eye(n);
       Iaux(:,j) = [];
       Iaux(j,:) = -1;
       
       C_j = [Ij,Iaux/sqrt(2)];
       
       % Calculate the expectation and multiply by e_j (extract component j) 
       [lInt,mu,~,~] = epmgp(m,K,C_j,lB,uB);
       Probs_y(j)    = exp(lInt);
       Int_y(j)      = mu(j)*exp(lInt);   % normalizing constant of the truncated probability 
    end
   
    % --------------------------------------------------------
    % ---- Compute the integral asociated to eta (second term)
    % ---------------------------------------------------------
    % Compute C_eta lB_eta,uB_eta
    C_eta = eye(n);
    lB_eta = repmat(eta,n,1);
    uB_eta = Inf(n,1);
    
    % Compute the (log) integral and take the exp
    [lInt_eta,~,~,~] = epmgp(m,K,C_eta,lB_eta,uB_eta);
    Int_eta = exp(lInt_eta);
    
    % ---- Sum-up
    e_min = sum(Int_y) + eta*Int_eta;

end

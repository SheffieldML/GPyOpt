% Javier Gonzalez
% 2015
%
% Given a kernel matrix L and a list of elements set returns a sample from a
% conditional k-DPP in dual form so it is faster to cumpute
%    L:     kernel matrix
%    q:     number of eigenventors
%    set:   index of the conditional elements. Integer numpy array containing the locations 
%            (starting in zero) relative to the rows of L.
%    k:     size of the sample from the DPP
 
function Y = sample_dual_conditional_dpp(L,q,set,k)
    [n,~] = size(L);
    n_set = length(set);
    
    % Calculate the kernel for the marginal    
    e                  = ones(1,n);
    e(set)             = 0;
    Id                 = diag(e);    
    L_aux              = inv(Id + L);
    L_aux(set,:)       = [];
    L_aux(:,set)       = [];
    L_minor            = inv(L_aux); 
    L_compset          = L_minor - eye(n-n_set);
    
    % truncate the kernel:obtain matrix B
    [V, D] = eigs(L_compset,q);
    B = V*sqrt(D); 
    
    % index to keep track of the original elements
    index_reduced = 1:n;
    index_reduced(set) = [];
    
    % Compute the sample from the marginal
    sample_conditional = sample_dual_dpp(B,decompose_kernel(B'*B),k-n_set);   
    
    % final sample that includes the set
    Y = [set,index_reduced(sample_conditional)];
    end
  
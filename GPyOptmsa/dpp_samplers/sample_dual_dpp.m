function Y = sample_dual_dpp(B,C,k)
% sample from a dual DPP (non-structured)
% B is the N x d feature matrix (L would be B*B', but is too big to work with)
% C is the decomposed covariance matrix, computed using:
%   C = decompose_kernel(B'*B);
% k is (optionally) the size of the set to return.

if ~exist('k','var')  
  % choose eigenvectors randomly
  D = C.D ./ (1+C.D);
  v = find(rand(length(D),1) <= D);
else
  % k-DPP
  v = sample_k(C.D,k);
end
k = length(v);
V = C.V(:,v);

% rescale eigenvectors so they normalize in the projected space
V = bsxfun(@times,V,1./sqrt(C.D(v)'));

% iterate
Y = zeros(k,1);
for i = k:-1:1

  % compute probabilities for each item
  P = sum((B * V).^2,2);
  P = P / sum(P);

  % choose a new item to include
  Y(i) = find(rand <= cumsum(P),1);
  
  % choose a vector to eliminate
  S = B(Y(i),:) * V;
  j = find(S,1);
  Vj = V(:,j);
  Sj = S(j);
  V = V(:,[1:j-1 j+1:end]);
  S = S(:,[1:j-1 j+1:end]);

  % update V
  V = V - bsxfun(@times,Vj,S/Sj);

  % orthogonalize in the projected space
  for a = 1:i-1
    for b = 1:a-1
      V(:,a) = V(:,a) - (V(:,a)'*C.M *V(:,b))*V(:,b);
    end
    V(:,a) = V(:,a) / sqrt(V(:,a)'*C.M*V(:,a));
  end

end

Y = sort(Y);

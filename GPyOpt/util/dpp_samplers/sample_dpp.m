function Y = sample_dpp(L,k)
% sample a set Y from a dpp.  L is a decomposed kernel, and k is (optionally)
% the size of the set to return.
  
if ~exist('k','var')  
  % choose eigenvectors randomly
  D = L.D ./ (1+L.D);
  v = find(rand(length(D),1) <= D);
else
  % k-DPP
  v = sample_k(L.D,k);
end
k = length(v);    
V = L.V(:,v);

% iterate
Y = zeros(k,1);
for i = k:-1:1
  
  % compute probabilities for each item
  P = sum(V.^2,2);
  P = P / sum(P);

  % choose a new item to include
  Y(i) = find(rand <= cumsum(P),1);

  % choose a vector to eliminate
  j = find(V(Y(i),:),1);
  Vj = V(:,j);
  V = V(:,[1:j-1 j+1:end]);

  % update V
  V = V - bsxfun(@times,Vj,V(Y(i),:)/Vj(Y(i)));

  % orthogonalize
  for a = 1:i-1
    for b = 1:a-1
      V(:,a) = V(:,a) - V(:,a)'*V(:,b)*V(:,b);
    end
    V(:,a) = V(:,a) / norm(V(:,a));
  end

end

Y = sort(Y);
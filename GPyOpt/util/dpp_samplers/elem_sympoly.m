function E = elem_sympoly(lambda,k)
% given a vector of lambdas and a maximum size k, determine the value of
% the elementary symmetric polynomials:
%   E(l+1,n+1) = sum_{J \subseteq 1..n,|J| = l} prod_{i \in J} lambda(i) 
  
  N = length(lambda);
  E = zeros(k+1,N+1);
  E(1,:) = 1;
  for l = (1:k)+1
    for n = (1:N)+1
      E(l,n) = E(l,n-1) + lambda(n-1)*E(l-1,n-1);
    end
  end

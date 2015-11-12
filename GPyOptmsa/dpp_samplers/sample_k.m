function S = sample_k(lambda,k)
% pick k lambdas according to p(S) \propto prod(lambda \in S)
  
  % compute elementary symmetric polynomials
  E = elem_sympoly(lambda,k);

  % iterate
  i = length(lambda);
  remaining = k;
  S = zeros(k,1);
  while remaining > 0

    % compute marginal of i given that we choose remaining values from 1:i
    if i == remaining
      marg = 1;
    else
      marg = lambda(i) * E(remaining,i) / E(remaining+1,i+1);
    end
    
    % sample marginal
    if rand < marg
      S(remaining) = i;
      remaining = remaining - 1;            
    end
    i = i-1;
  end

function X = genmult(A,B)
% generalized matrix multiply.  
% if A is N x M, and B is 1 x 1 x ... x 1 x M x K1 x K2 x ...
% then X is 1 x 1 x ... x 1 x N x K1 x K2 x ...
% where X(1,1,...,1,:,i1,i2,...) = A * B(1,1,...,1,:,i1,i2,...)
  
  % strip off leading singletons
  [B,shifts] = shiftdim(B);
  
  % make B 2-d
  Bsize = size(B);
  B = reshape(B,Bsize(1),[]);
  
  % compute product
  X = A*B;
  
  % repair original dimensions
  X = reshape(X,[size(A,1) Bsize(2:end)]);  
  X = shiftdim(X,-shifts);
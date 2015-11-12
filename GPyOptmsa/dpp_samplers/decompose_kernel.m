function L = decompose_kernel(M)
  L.M = M;
  [V,D] = eig(M);
  L.V = real(V);
  L.D = real(diag(D));
function result = bp(M,mode,V,assign)
% run BP on a sequence model
% 
% - M is a sequence model with fields:
%     T = length of sequence
%     N = number of labels (all nodes assumed to take same labels)
%     A = N x N sparse edge potential matrix; A(i,j) is potential from label i to j
%     Q = 1 x N node potential vector (applies to all except (possibly) first node)
%     Q1 = 1 x N initial node quality vector (if omitted, uses Q)
%     G = N x D similarity features (only needed for 2nd-order modes)
% 
% - mode specifies the semiring/result, and can be:
%
%   'partition' to compute the partition function, result is scalar.
%
%   'marginals' to compute marginals, result is T x N.
% 
%   'sample' to generate a sample, result is T x 1.
%
%   'marginals2' to compute second-order marginals, result is T x N:
%      sum_{y ~ y_i} p(y) sum_{j=1}^r (V(:,j) . G(y))^2
%
%   'sample2' to generate a second-order sample, result is T x 1:
%      P(y) ~ p(y) sum_{j=1}^r (V(j,:) . G(y))^2
%
%   'covariance' to compute similarity feature covariance, result is D x D
%
% - V is a D x r set of r weight vectors on similarity features, needed 
%   only for modes 'marginals2' and 'sample2'.
%
% - assign is an optional T x 1 partial assignment vector.  nonzero values 
%   are visible labels, zeros are hidden labels.
  
  if ~exist('assign','var') || isempty(assign)
    assign = zeros(M.T,1);
  end
  assert(length(assign) == M.T);
  
  if ~isfield(M,'Q1')
    M.Q1 = M.Q;
  end


  %% forward pass
  
  % init
  switch mode

   case {'partition','marginals','sample'}
    Fq = zeros(M.T,M.N);
        
   case {'marginals2','sample2'}
    r = size(V,2);    
    Fq = zeros(M.T,M.N);
    Ff = zeros(M.T,M.N,r);
    Fff = zeros(M.T,M.N,r);

    % pre-compute, f = 1 x N x r
    f = shiftdim(M.G*V,-1);
    
   case 'covariance'    
    % to save memory, keep only the most recent message
    Fq = zeros(M.N,1);
    Fg = zeros(M.N,size(M.G,2));
    Fc = zeros(M.N,size(M.G,2),size(M.G,2));
    
    % pre-compute, gg = N x D x D
    gg = bsxfun(@times,M.G,permute(M.G,[1,3,2]));

  end
  
  % first messages
  switch mode

   case {'partition','marginals','sample'}
    if assign(1)
      Fq(1,assign(1)) = 1;
    else
      Fq(1,:) = M.Q1;
    end
    
   case {'marginals2','sample2'}
    if assign(1)
      Fq(1,assign(1)) = M.Q1(assign(1));
    else
      Fq(1,:) = M.Q1;
    end
    Ff(1,:,:) = bsxfun(@times,Fq(1,:),f);
    Fff(1,:,:) = bsxfun(@times,Fq(1,:),f.^2);

   case 'covariance'
    if assign(1)
      Fq(assign(1)) = 1;
    else
      Fq = M.Q1;
    end
    Fg = bsxfun(@times,Fq(1,:)',M.G);
    Fc = bsxfun(@times,Fq(1,:)',gg);

  end
  
  % go
  for t = 2:M.T

    if assign(t)
      notallowed = setdiff(1:M.N,assign(t));
    else
      notallowed = [];
    end
    
    switch mode

     case {'partition','marginals','sample'}
      Fq(t,:) = M.Q .* (Fq(t-1,:) * M.A);
      Fq(t,notallowed) = 0;
            
     case {'marginals2','sample2'}
      AFf = genmult(M.A',Ff(t-1,:,:));                  
      AFff = genmult(M.A',Fff(t-1,:,:));

      Fq(t,:) = M.Q .* (Fq(t-1,:) * M.A);
      Ff(t,:,:) = bsxfun(@times,Fq(t,:),f) ...
          + bsxfun(@times,M.Q,AFf);                   
      Fff(t,:,:) = bsxfun(@times,Fq(t,:),f.^2) ...
          + bsxfun(@times, M.Q, 2 * f .* AFf + AFff);

      Fq(t,notallowed) = 0;
      Ff(t,notallowed,:) = 0;
      Fff(t,notallowed,:) = 0;

     case 'covariance'
      AFg = M.A' * Fg;
      Across = bsxfun(@times,AFg,permute(M.G,[1 3 2]));

      Fq = M.Q .* (Fq * M.A);
      Fg = bsxfun(@times,Fq',M.G) + bsxfun(@times,M.Q',AFg);
      Fc = bsxfun(@times,Fq',gg) ...
          + bsxfun(@times,M.Q', ...
                   Across + permute(Across,[1 3 2]) ...
                   + genmult(M.A',Fc));
      
      Fq(notallowed) = 0;
      Fg(notallowed,:) = 0;
      Fc(notallowed,:,:) = 0;
    
    end
  end

  %% backward pass
  
  % init
  switch mode

   case 'marginals'
    Bq = zeros(M.T,M.N);
   
   case 'sample'
    Bq = zeros(M.T,M.N);
    Y = zeros(M.T,1);
    
   case 'marginals2'
    Bq = zeros(M.T,M.N);
    Bf = zeros(M.T,M.N,r);
    Bff = zeros(M.T,M.N,r);

   case 'sample2'
    Bq = zeros(M.T,M.N);
    Bf = zeros(M.T,M.N,r);
    Bff = zeros(M.T,M.N,r);
    Y = zeros(M.T,1);
    
  end
  
  % first messages
  switch mode

   case 'marginals'
    if assign(M.T)
      Bq(M.T,assign(M.T)) = 1;
    else
      Bq(M.T,:) = ones(1,M.N);
    end
   
   case 'sample'
    if assign(M.T)
      Y(M.T) = assign(M.T);
    else
      % sample node
      dist = Fq(M.T,:);
      Y(M.T) = find(rand <= cumsum(dist) / sum(dist),1);      
    end
    Bq(M.T,Y(M.T)) = 1;
        
   case 'marginals2'
    if assign(M.T)
      Bq(M.T,assign(M.T)) = 1;
    else
      Bq(M.T,:) = ones(1,M.N);
    end
    Bf(M.T,:,:) = zeros(1,M.N,r);
    Bff(M.T,:,:) = zeros(1,M.N,r);

   case 'sample2'
    if assign(M.T)
      Y(M.T) = assign(M.T);
    else
      % sample node
      dist = sum(Fff(M.T,:,:),3);
      Y(M.T) = find(rand <= cumsum(dist) / sum(dist),1);
    end
    Bq(M.T,Y(M.T)) = 1;
    Bf(M.T,:,:) = zeros(1,M.N,r);
    Bff(M.T,:,:) = zeros(1,M.N,r);    
    
  end
  
  % go
  for t = M.T-1:-1:1
    if assign(t)
      notallowed = setdiff(1:M.N,assign(t));
    else
      notallowed = [];
    end
    
    switch mode

     case 'marginals'
      Bq(t,:) = M.A * (M.Q .* Bq(t+1,:))';
      Bq(t,notallowed) = 0;
      
     case 'sample'
      Bq(t,:) = M.A * (M.Q .* Bq(t+1,:))';
      
      if assign(t)
        Y(t) = assign(t);
      else
        % sample node
        dist = Fq(t,:) .* Bq(t,:);
        Y(t) = find(rand <= cumsum(dist) / sum(dist),1);
      end
      notallowed = setdiff(1:M.N,Y(t));

      Bq(t,notallowed) = 0;
     
     case 'marginals2'
      Bq(t,:) = M.A * (M.Q .* Bq(t+1,:))';   
      Bf(t,:,:) = genmult(M.A,bsxfun(@times,M.Q .* Bq(t+1,:),f) ...
                          + bsxfun(@times,M.Q,Bf(t+1,:,:)));
      Bff(t,:,:) = genmult(M.A,bsxfun(@times,M.Q .* Bq(t+1,:),f.^2) ...
                           + bsxfun(@times,M.Q, ...
                                    Bff(t+1,:,:) + 2*f.*Bf(t+1,:,:)));

      Bq(t,notallowed) = 0;
      Bf(t,notallowed,:) = 0;
      Bff(t,notallowed,:) = 0;

     case 'sample2'
      Bq(t,:) = M.A * (M.Q .* Bq(t+1,:))';
      Bf(t,:,:) = genmult(M.A,bsxfun(@times,M.Q .* Bq(t+1,:),f) ...
                          + bsxfun(@times,M.Q,Bf(t+1,:,:)));
      Bff(t,:,:) = genmult(M.A,bsxfun(@times,M.Q .* Bq(t+1,:),f.^2) ...
                           + bsxfun(@times,M.Q, ...
                                    Bff(t+1,:,:) + 2*f.*Bf(t+1,:,:)));

      if assign(t)
        Y(t) = assign(t);
      else
        % sample node
        dist = sum(bsxfun(@times,Fq(t,:),Bff(t,:,:)) + ...
                   bsxfun(@times,Bq(t,:),Fff(t,:,:)) + ...
                   2*Bf(t,:,:).*Ff(t,:,:),3);
        Y(t) = find(rand <= cumsum(dist) / sum(dist),1);
      end
      notallowed = setdiff(1:M.N,Y(t));
      
      Bq(t,notallowed) = 0;
      Bf(t,notallowed,:) = 0;
      Bff(t,notallowed,:) = 0;

    end
  end
  
  % set result
  switch  mode
   case 'partition'
    result = sum(Fq(M.T,:));
    
   case 'marginals'
    result = Fq .* Bq;

   case 'sample'
    result = Y;

   case 'marginals2'
    result = sum(bsxfun(@times,Fq,Bff) + bsxfun(@times,Bq,Fff) + 2*Bf.*Ff,3);

   case 'sample2'
    result = Y;

   case 'covariance'
    result = shiftdim(sum(Fc(:,:,:)));
    
  end
  

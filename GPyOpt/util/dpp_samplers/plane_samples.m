%% compare dpp/poisson samples of points in the plane

% config
n = 60;      % grid dimension, N = n^2
sigma = 0.1; % kernel width

% choose a grid of points
[x y] = meshgrid((1:n)/n);

% gaussian kernel
L = exp(- (bsxfun(@minus,x(:),x(:)').^2 + ...
           bsxfun(@minus,y(:),y(:)').^2) / sigma^2);

% sample
dpp_sample = sample_dpp(decompose_kernel(L));
ind_sample = randsample(n*n,length(dpp_sample));
  
% plot
subplot(1,2,1);
plot(x(dpp_sample),y(dpp_sample),'b.');
axis([0 1.02 0 1.02]);
axis square;
set(gca,'YTick',[]);
set(gca,'XTick',[]);
xlabel('DPP');

subplot(1,2,2);
plot(x(ind_sample),y(ind_sample),'r.');
axis([0 1.02 0 1.02]);
axis square;
set(gca,'YTick',[]);
set(gca,'XTick',[]);
xlabel('Independent');

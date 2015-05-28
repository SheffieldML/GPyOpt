% Javier Gonzalez
% 2015
%
% test and compare samples from a k-dpp, a conditional k-dpp, and a 
% conditional k-dpp in dual form

% config
n = 60;      % grid dimension, N = n^2
sigma = 0.1; % kernel width

% choose a grid of points
[x y] = meshgrid((1:n)/n);

% gaussian kernel
L = exp(- (bsxfun(@minus,x(:),x(:)').^2 + ...
           bsxfun(@minus,y(:),y(:)').^2) / sigma^2);

set = [10,100,500,750,1500,3500];
k   = 50;                           % number of elements in the sample
q   = 100;                          % effective dimensions used

% sample form k-DPP
dpp_sample             = sample_dpp(decompose_kernel(L),k);

% conditional sample from DPP 
t_standard = cputime;
dpp_sample_conditional = sample_conditional_dpp(L,set,k);
t_standard = cputime - t_standard; 

% conditional sample from DPP in dual form
t_dual = cputime;
dpp_dual_sample_conditional = sample_dual_conditional_dpp(L,q,set,k);
t_dual = cputime - t_dual ;


% plot
subplot(1,3,1);
plot(x(dpp_sample),y(dpp_sample),'b.');
axis([0 1.02 0 1.02]);
xlabel('DPP');

subplot(1,3,2);
plot(x(set),y(set),'r*');
hold on;
plot(x(dpp_sample_conditional),y(dpp_sample_conditional),'r.');
axis([0 1.02 0 1.02]);
xlabel('Conditional DPP');

subplot(1,3,3);
plot(x(set),y(set),'r*');
hold on;
plot(x(dpp_dual_sample_conditional),y(dpp_dual_sample_conditional),'r.');
axis([0 1.02 0 1.02]);
xlabel('Conditional Dual DPP');




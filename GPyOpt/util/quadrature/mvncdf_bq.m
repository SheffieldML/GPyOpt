function [ m_Z, sd_Z, data, log_sd_Z ] = mvncdf_bq( l, u, mu, Sigma, opt )
% Bayesian quadrature for Gaussian integration. Our domain has dimension N.
%
% INPUTS
% mu: mean of Gaussian (N * 1)
% Sigma: covariance of Gaussian (N * N)
% l: vector of lower bounds (N * 1), if missing, assumed to be all -inf
% u: vector of upper bounds (N * 1), if missing, assumed to be all inf
% opt: options (see below)
% in particular, if opt.data is supplied, the locations for the Gaussian
% convolution observations are supplied.
% opt.data(i).m represents the mean of a Gaussian, 
% opt.data(i).m V the diagonal of its diagonal covariance.
% i takes as many sequential values as you want to give.
%
% OUTPUTS
% m_Z: mean of Gaussian integral
% sd_Z: standard deviation of Gaussian integral
%
% Michael Osborne 2012

start_time = cputime;
N = size(mu, 1);

if nargin < 3 || isempty(l)
   l = -inf(N, 1);
end
if nargin < 4 || isempty(u)
   u = inf(N, 1);
end
if nargin < 5
   opt = struct();
end

% stop inf * 0 situations arising below.
u = min(u, 1/eps);
l = min(l, 1/eps);
u = max(u, -1/eps);
l = max(l, -1/eps);


% Set unspecified fields to default values.
default_opt = struct('total_time', 300, ...
                    'data', []);
opt = set_defaults( opt, default_opt );


% Define terms required to evaluate the covariances between any pair of
% convolution observations, and the covariances between the target
% hyper-rectangle and any convolution observation.
% =========================================================================

% covariance function of latent function is mvnpdf(x, x', diag(Om));
Om = diag(Sigma);

% prior is mvnpdf(x, mu, diag(L)); this defines the envelope outside which
% we expect our Gaussian integrand to be zero
cvx_begin sdp
variable L(N,N) diagonal
minimize(trace(L))
L >= Sigma
cvx_end
L = diag(L);

% We're about to compute the product of the bivariate Gaussian cdfs,
% sum_log_mvncdf_lu_N,  that constitutes the self-variance of the target
% hyper-rectangle.

% The two off-diagonal elements of the 2*2 covariance matrix over x1(d) and
% x2(d) (where x1 and x2 are arbitrary points in the domain of integration)
% are equal to M_offdiag(d)
M_offdiag = L.^2./(2*L + Om);
% The two on-diagonal elements of the 2*2 covariance matrix over x1(d) and
% x2(d) (where x1 and x2 are arbitrary points in the domain of integration)
% are equal to M_ondiag(d)
M_ondiag = L - M_offdiag;

sum_log_mvncdf_lu_N = 0;
for d = 1:N;
    
    % both variables x1(d) and x2(d) have mu(d) as their mean, l(d) as
    % their lower limit and u(d) as their upper limit
    l_d = [l(d); l(d)];
    u_d = [u(d); u(d)];
    mean_d = [mu(d); mu(d)];
    cov_d = [M_ondiag(d), M_offdiag(d);
            M_offdiag(d), M_ondiag(d)];

    sum_log_mvncdf_lu_N = sum_log_mvncdf_lu_N + ...
        log(mvncdf(l_d, u_d, mean_d, cov_d));
end
% log_variance is the log of the squared output scale, chosen so that K_tt
% is exactly one. Note that we also drop the K_const = N(0,0,2*L + Om)
% factor from all covariances; this term gets lumped in with the output
% scale.
log_variance = - sum_log_mvncdf_lu_N;

% K is always a covariance matrix. Below, we use the subscripts:
% t: target hyper-rectangle
% g: new gaussian convolution
% d: all gaussian convolutions

% variance of target hyper-rectangle; we've scaled so that this is one.
K_t = 1;

% Take or read in data
% =========================================================================

% Initialise R_d, the cholesky factor of the covariance matrix over data
R_d = nan(0, 0);
% Initialise Da_d = inv(R_d') * (convolution observations);
Da_d = nan(0, 1);
% Initialise S_dt = inv(R_d') * K_td';
S_dt = nan(0, 1);

% active_data_selection = true implies that we intelligently select
% observations. If false, we simply read in data from inputs. 
active_data_selection = isempty(opt.data);

% initialise the structure that will store our Gaussian convolution
% observations in it. m represents the mean of such a Gaussian, V the
% diagonal of its diagonal covariance, and conv the actual convolution
% observation. 
data = []; 
    
if active_data_selection
    
    start_time = cputime;
    while cputime - start_time < opt.total_time
        % add new observation
    
        if rem(d, 10) == 0
            fprintf('\n%g',d);
        else
            fprintf('.');
        end
        
        [m_g, V_g] = min_variance(mu, Sigma, ...
            M_ondiag, M_offdiag, l, u, log_variance, ...
            R_d, Da_d, S_dt, data);
        
        [R_d, Da_d, S_dt, data] = ...
            add_new_datum(m_g, V_g, mu, Sigma, ...
            M_ondiag, M_offdiag, l, u, log_variance, ...
            R_d, Da_d, S_dt, data);
    end
    
else
    full_data = opt.data;
    
    for d = 1:numel(full_data)
    % add new observation
    
        if rem(d, 10) == 0
            fprintf('\n%g',d);
        else
            fprintf('.');
        end
        
        m_g = full_data(d).m;
        V_g = full_data(d).V;
        
        [R_d, Da_d, S_dt, data] = ...
            add_new_datum(m_g, V_g, mu, Sigma, ...
            M_ondiag, M_offdiag, l, u, log_variance, ...
            R_d, Da_d, S_dt, data);
    end
end

% Compute final prediction
% =========================================================================
    
[m_Z, var_Z] = predict(K_t, Da_d, S_dt);
sd_Z = sqrt(var_Z);

% Above, we chose an absurdly large sqd output scale (exp(log_variance)) so
% as to make things more numerically stable. Now we have to scale to
% correct for our scaling above; the mean is unaffected, given our
% observations are noiseless, and the variance is affected only by a simple
% multiplicative factor.

% this is the volume of the bounding Gaussian enclosed within the bounds
realistic_log_sd = sum(truncNormMoments(l, u, mu, L));

log_sd_Z = realistic_log_sd + log(sd_Z);
sd_Z = exp(log_sd_Z);

end

function [R_d, Da_d, S_dt, data, ...
    T_dt, dK_td_dm_g, dK_d_dm_g, dK_td_dV_g, dK_d_dV_g] = ...
    add_new_datum(m_g, V_g, mu, Sigma, ...
    M_ondiag, M_offdiag, l, u, log_variance, ...
    R_d, Da_d, S_dt, data)
% Update to include new convolution, NB: d includes g
% m_g: mean of new Gaussian convolution (n * 1) 
% V_g: diagonal of (diagonal) covariance of 
%           new Gaussian convolution (n * 1)
% optionally also output quantities sufficient to calculate gradient of
% variance wrt m_g and V_g

var_and_grad_only = nargout > 4;
% output quantities sufficient to calculate gradient of variance wrt 
% m_g and V_g

% update the number of data
num_data = numel(data) + 1;

% add new convolution observation to data structure
% =========================================================================
  
data(num_data).m = m_g;
data(num_data).V = V_g;
if ~var_and_grad_only
    % mvnpdf requires a cholesky factorisation, which scales as O(N^3)
    data(num_data).conv = mvnpdf(m_g, mu, diag(V_g) + Sigma);
else
    data(num_data).conv = nan;
end

N = length(mu);

% compute new elements of covariance matrix over data, K_d
% =========================================================================
  
log_K_gd = log_variance * ones(1, num_data);
for i = 1:num_data

    m_di = data(i).m;
    V_di = data(i).V;
    
    % K_gd(i) is a product of bivariate Gaussians, one for each dimension
    for d = 1:N;
        
        val_d = [m_g(d); m_di(d)];
        mean_d = [mu(d); mu(d)];
        
        cov_d = [M_ondiag(d) + V_g(d), M_offdiag(d);
                M_offdiag(d), M_ondiag(d) + V_di(d)];
        
        log_K_gd(i) = log_K_gd(i) + ...
            logmvnpdf(val_d, mean_d, cov_d);
    end
end
K_gd = exp(log_K_gd);

% update cholesky factor R_d of covariance matrix over data, K_d
% =========================================================================
  
old_R_d = R_d; % need to store this to update Da_d and S_dt, below
K_d = [nan(num_data-1), K_gd(1:end-1)'; 
        K_gd(1:end-1), improve_covariance_conditioning(K_gd(end))];
R_d = updatechol(K_d, old_R_d, num_data);

% update product Da_d = inv(R_d') * [data(:).conv]';
% =========================================================================
  
Da_d = updatedatahalf(R_d, vertcat(data(:).conv), Da_d, old_R_d, num_data);

% compute new elements of covariance vector between target and data
% =========================================================================
  
% p stands for 'posterior'
pm_g = mu + M_offdiag .* (M_ondiag + V_g).^-1 .* (m_g - mu);
pV_g = M_ondiag - M_offdiag .* (M_ondiag + V_g).^-1 .* M_offdiag;
pSD_g = sqrt(pV_g);

data(num_data).pm = pm_g;
data(num_data).pV = pV_g;

sumlog_K_tg_normpdfs = sum(lognormpdf(m_g, mu, sqrt(M_ondiag + V_g)));
log_K_tg_normcdfs = truncNormMoments(l, u, pm_g, pSD_g); % this computes log moments
log_K_tg = log_variance + ...
             sumlog_K_tg_normpdfs + ...
            sum(log_K_tg_normcdfs);
K_tg = exp(log_K_tg);

% zeros in K_td are not used in the process of generating S_dt and help
% with the computation of gradients below
K_td = [zeros(1, num_data-1), K_tg];

% update product S_dt = inv(R_d') * K_td';
% =========================================================================
  
S_dt = updatedatahalf(R_d, K_td', S_dt, old_R_d, num_data);

if var_and_grad_only

    % determine T_dt = inv(K_d') * K_td';
    % =====================================================================

    T_dt = R_d \ S_dt;
    
    % stack turns the rows of a matrix into stacked plates
    stack = @(mat) reshape(full(mat), 1, size(mat, 2), N);
    
    % quad is the quadratic form -0.5 * (mu - m_g)^2/(M_ondiag + V_g)
    
    % first, compute dK_td_dm_g (the gradient of dK_td wrt m_g)
    dquad_m_g_stack = stack((mu - m_g) ./ (M_ondiag + V_g));
    dpm_g_dm_g = ...
        - M_offdiag .* (M_ondiag + V_g).^-2 .* (m_g - mu);
    dpV_g_dm_g = ...
          M_offdiag .* (M_ondiag + V_g).^-2 .* M_offdiag; 
    
    dK_td_dm_g = bsxfun(@times, dquad_m_g_stack, K_td) ...
         + 0.5 * bsxfun(@times, K_td, ...
        exp(- stack(log_K_tg_normcdfs) .* stack( ...
        normpdf(u, pm_g, pSD_g) ...
            .* (2 * pV_g .* dpm_g_dm_g + (u - pm_g) .* dpV_g_dm_g) ...
        - normpdf(l, pm_g, pSD_g) ...
            .* (2 * pV_g .* dpm_g_dm_g + (l - pm_g) .* dpV_g_dm_g))));
   
    % second, compute dK_d_dm_g (the gradient of dK_d wrt m_g)

    dK_gd_dm_g = bsxfun(@times, K_gd, stack( ...
        (bsxfun(@minus, horzcat(data(:).pm), m_g)) ...
        ./ (bsxfun(@plus, horzcat(data(:).pV), V_g)) ));
    
    % the last datum is g (last element of d is g)
    
    det_M_plus_V_g = (M_ondiag + V_g).^2 + M_offdiag.^2;
    dK_g_dm_g = K_gd(:, end) .* stack( ...
        2 * (mu - m_g) .* (M_ondiag + V_g - M_offdiag) ./ det_M_plus_V_g );
    % reminder: M_offdiag = L.^2./(2*L + Om);
    %           M_ondiag = L - M_offdiag;
    
    dK_gd_dm_g(:, end, :) = dK_g_dm_g;
    
    dK_d_dm_g = [zeros(num_data-1, num_data-1, N), ...
                    tr(dK_gd_dm_g(:, 1:end-1, :)); % d includes g
                dK_gd_dm_g];
    
    % third, compute dK_td_dV_g (the gradient of dK_td wrt V_g)

    dquad_V_g_stack = stack(0.5 * ((mu - m_g).^2 ./ (M_ondiag + V_g) - 1) ...
                       ./ (M_ondiag + V_g));
    dpm_g_dV_g = ...
        - M_offdiag .* (M_ondiag + V_g).^-2 .* (m_g - mu);
    dpV_g_dV_g = ...
          M_offdiag .* (M_ondiag + V_g).^-2 .* M_offdiag; 
      
    dK_td_dV_g = bsxfun(@times, dquad_V_g_stack, K_td) ...
        + 0.5 * bsxfun(@times, K_td, ...
        exp(- stack(log_K_tg_normcdfs) .* stack( ...
        normpdf(u, pm_g, pSD_g) ...
            .* (2 * pV_g .* dpm_g_dV_g + (u - pm_g) .* dpV_g_dV_g) ...
        - normpdf(u, pm_g, pSD_g) ...
            .* (2 * pV_g .* dpm_g_dV_g + (l - pm_g) .* dpV_g_dV_g))));


    % fourth, compute dK_d_dV_g (the gradient of dK_d wrt V_g)

    dK_gd_dV_g = 0.5 * bsxfun(@times, K_gd, stack( ...
        ( bsxfun(@minus, horzcat(data(:).pm), m_g).^2 ...
            - horzcat(data(:).pV) ...
        ) ./  horzcat(data(:).pV).^2 ));
    
    % the last datum is g (last element of d is g)
    
    dK_g_dV_g = -K_gd(:, end) .* stack( ...
        (M_ondiag + V_g) ./ det_M_plus_V_g ...
        + 2 * (mu - m_g).^2 ./ (M_ondiag + M_offdiag + V_g).^2);
    % reminder: M_offdiag = L.^2./(2*L + Om);
    %           M_ondiag = L - M_offdiag;
    
    dK_gd_dV_g(:, end, :) = dK_g_dV_g;
    
    dK_d_dV_g = [zeros(num_data-1, num_data-1, N), ...
                    tr(dK_gd_dV_g(:, 1:end-1, :)); % d includes g
                dK_gd_dV_g];
end

end

function [m, var, dvar_dm_g, dvar_dV_g] = predict(K_t, Da_d, S_dt, T_dt, ...
    dK_td_dm_g, dK_d_dm_g, dK_td_dV_g, dK_d_dV_g)
% returns the mean m and standard deviation sd in the target Gaussian
% integral. If supplied with appropriate additional gradients, also returns
% the gradients of the variance with respect to mean of new observation,
% dvar_dm, and its vector of variances, dvar_dV.

% gp posterior variance
var = K_t - S_dt' * S_dt;

if nargout < 3
    % gp posterior mean
    m = S_dt' * Da_d;
else
    % function called purely to provide gradient of variance
    m = nan;
    
    dvar_dm_g = -2 * prod3(dK_td_dm_g, T_dt) ...
                + prod3(T_dt', prod3(dK_d_dm_g, T_dt));
    
    dvar_dV_g = -2 * prod3(dK_td_dV_g, T_dt) ...
                + prod3(T_dt', prod3(dK_d_dV_g, T_dt));
end

end

function [m, V] = min_variance(mu, Sigma, ...
    K_t, M_ondiag, M_offdiag, l, u, log_variance, ...
    R_d, Da_d, S_dt, data)
% note that we can compute the variance after any new observation without
% requiring an expectation to be computed over the potential convolutions
% returned by the observation: the variance of a GP is independent of the
% function values.

% trial
V_g = rand;
m_g = rand;

[~, ~, S_dt, ~, ...
    T_dt, dK_td_dm_g, dK_d_dm_g, dK_td_dV_g, dK_d_dV_g] = ...
    add_new_datum(m_g, V_g, mu, Sigma, ...
    M_ondiag, M_offdiag, l, u, log_variance, ...
    R_d, Da_d, S_dt, data);

[~, var, dvar_dm_g, dvar_dV_g] = predict(K_t, nan, S_dt, T_dt, ...
    dK_td_dm_g, dK_d_dm_g, dK_td_dV_g, dK_d_dV_g);

end


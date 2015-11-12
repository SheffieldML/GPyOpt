function draw_from_linearisation

% beta controls how wide the error bars are for large likelihoods
beta = 5;
% alpha sets how wide the error bars are for small likelihoods
alpha = 1/100;

% NB: results seem fairly insensitive to selection of alpha (sigh). beta
% has desired effect.

% define observed likelihood, lik, & locations of predictants
% ====================================

% multiply fixed likelihood function by this constant
const = 10^(5*(rand-.5));

lik = rand(6,1)/const;
%lik = ([0.1;0.25;0.1;0.15;0.18;0.15;0.48;0.05])/const;
n = length(lik);
x = linspace(0,10,n)';

n_st = 1000;
xst = linspace(min(x)-10, max(x)+10, n_st)';

% define map f(tlik) = lik
% ====================================

mn = min(lik);
mx = max(lik);

% Maximum likelihood approach to finding map

f_h = @(tlik, theta) mn * exp(theta(1) * tlik.^2 + theta(2) * tlik);
df_h = @(tlik, theta) f_h(tlik, theta) .* (2 .* theta(1) * tlik + theta(2));
ddf_h = @(tlik, theta) ...
    f_h(tlik, theta) .* (2 .* theta(1)) ...
    + df_h(tlik, theta) .* (2 .* theta(1) * tlik + theta(2));
invf_h = @(lik, theta) 1/(2*theta(1)) * (-theta(2) + ...
    sqrt(4 * theta(1) * log(lik/mn) + theta(2)^2)...
                                );                         
options.MaxFunEvals = 1000;
[logth,min_obj] = fminunc(@(logth) objective(logth, f_h, df_h, mn, mx, ...
    alpha, beta), ...
    zeros(1,3), options);
min_obj
theta = exp(logth);

f = @(tlik) f_h(tlik,theta);
df = @(tlik) df_h(tlik, theta);
ddf = @(tlik) ddf_h(tlik, theta);
invf = @(lik) invf_h(lik, theta);

figure(3)
clf
n_tliks = 1000;
tliks = linspace(0, theta(3), n_tliks);
plot(tliks,f(tliks),'b');
hold on
%plot(tliks,df(tliks),'r');

xlabel 'transformed likelihood'
ylabel 'likelihood'

% define gp over inv-transformed (e.g. log-) likelihoods
% ====================================

invf_lik = invf(lik);

% gp covariance hypers
w = 1;
h = std(invf_lik);
sigma = eps;
mu = min(invf_lik);

% define covariance function
fK = @(x, xd) h^2 .* exp(-0.5*(x - xd).^2 / w^2);
K = @(x, xd) bsxfun(fK, x, xd');

% Gram matrix
V = K(x, x) + eye(n)*sigma^2;

% GP posterior for tlik
m_tlik = mu + K(xst, x) * (V\(invf_lik-mu));
C_tlik = K(xst, xst) - K(xst, x) * (V\K(x, xst));
sd_tlik = sqrt(diag(C_tlik));

figure(1)
clf
gp_plot(xst, m_tlik, sd_tlik, x, invf(lik));
ylabel(sprintf('transformed\n likelihood'))

% define linearisation, lik = a * tlik + c
% ====================================


% exact linearisation: unusable in practice
a_exact = df(m_tlik);
c_exact = f(m_tlik) - a_exact .* m_tlik;

% approximate linearisation
best_tlik = mu;
a = df(best_tlik) + (m_tlik - best_tlik) .* ddf(best_tlik);
c = f(m_tlik) - a .* m_tlik;

figure(3)
for i = round(linspace(0.2*n_st, 0.8*n_st, 5))
    
    x_vals = linspace(m_tlik(i) - sd_tlik(i), m_tlik(i) + sd_tlik(i), 100);
    y_vals = a(i) * x_vals + c(i);
    y_vals_exact = a_exact(i) * x_vals + c_exact(i);
    
    plot(x_vals, y_vals_exact, 'k')
    plot(x_vals, y_vals, 'r')

    
end

% gp over likelihood
% ====================================

m_lik = diag(a) * m_tlik + c;
C_lik =  diag(a) * C_tlik * diag(a);
sd_lik = sqrt(diag(C_lik));

figure(2)
clf
gp_plot(xst, m_lik, sd_lik, x, lik);
ylabel 'likelihood'
%plot(xst, a,'g')

function [f, df] = objective(logth, f_h, df_h, mn, mx, alpha, beta)

% maximum likelihood (or least squares) objective for our three constraints:
% exp(logth(3)) * df_h(0,exp(logth)) == alpha .* (mx - mn)
% f_h(exp(logth(3)), exp(logth)) == mx
% exp(logth(3)) * df_h(exp(logth(3)), exp(logth)) == beta .* (mx - mn)

f = (exp(logth(3)) * df_h(0,exp(logth)) - alpha .* (mx - mn)).^2 + ...
    (f_h(exp(logth(3)), exp(logth)) - mx).^2 + ...
    (exp(logth(3)) * df_h(exp(logth(3)), exp(logth)) - beta .* (mx - mn)).^2;

df = nan(3,1);

% these derivatives should ideally be keyed off input dtheta_f_h etc. input
% functions, but currently they are not
df(1) = ...
   2 .*exp(exp(logth(2)+logth(3))+exp(logth(1)+2 .*logth(3))+logth(1)+2 .*logth(3)) .* mn .* (exp(exp(logth(2)+logth(3))+exp(logth(1)+2 .*logth(3))) .* mn-mx+(2+exp(logth(2)+logth(3))+2 .*exp(logth(1)+2 .*logth(3))) .* (exp(exp(logth(2)+logth(3))+exp(logth(1)+2 .*logth(3))+logth(2)+logth(3)) .* mn+2 .*exp(exp(logth(2)+logth(3))+exp(logth(1)+2 .*logth(3))+logth(1)+2 .*logth(3)) .* mn+(mn-mx) .* beta));
        
        
df(2) = ...
2 * exp(exp(logth(2))+exp(logth(3))) .* mn .* (exp(exp(logth(2))+exp(logth(3))) .* mn+exp(exp(exp(logth(2))+exp(logth(3)))+exp(exp(logth(1))+2 * exp(logth(3)))) .* (exp(exp(exp(logth(2))+exp(logth(3)))+exp(exp(logth(1))+2 * exp(logth(3)))) .* mn-mx)+(mn-mx) * alpha+exp(exp(exp(logth(2))+exp(logth(3)))+exp(exp(logth(1))+2 * exp(logth(3)))) .* (1+exp(exp(logth(2))+exp(logth(3)))+2 * exp(exp(logth(1))+2 * exp(logth(3)))) .* (exp(exp(exp(logth(2))+exp(logth(3)))+exp(exp(logth(1))+2 * exp(logth(3)))+exp(logth(2))+exp(logth(3))) .* mn+2 * exp(exp(exp(logth(2))+exp(logth(3)))+exp(exp(logth(1))+2 * exp(logth(3)))+exp(logth(1))+2 * exp(logth(3))) .* mn+(mn-mx) * beta));

df(3) = ... 
    2 * exp(logth(3)) .* mn .* (exp(exp(exp(logth(2))+exp(logth(3)))+exp(exp(logth(1))+2 * exp(logth(3)))) .* (exp(logth(2))+2 * exp(exp(logth(1))+exp(logth(3)))) .* (exp(exp(exp(logth(2))+exp(logth(3)))+exp(exp(logth(1))+2 * exp(logth(3)))) * mn-mx)+exp(logth(2)) .* (exp(exp(logth(2))+exp(logth(3))) * mn+(mn-mx) \[alpha])+exp(exp(exp(logth(2))+exp(logth(3)))+exp(exp(logth(1))+2 * exp(logth(3)))) .* (exp(logth(2))+4 * exp(exp(logth(1))+exp(logth(2))+2 * exp(logth(3)))+4 * exp(2 * exp(logth(1))+3 * exp(logth(3)))+exp(logth(3)) .* (4 * exp(logth(1))+exp(2 * exp(logth(2))))) .* (exp(exp(exp(logth(2))+exp(logth(3)))+exp(exp(logth(1))+2 * exp(logth(3)))+exp(logth(2))+exp(logth(3))) * mn+2 * exp(exp(exp(logth(2))+exp(logth(3)))+exp(exp(logth(1))+2 * exp(logth(3)))+exp(logth(1))+2 * exp(logth(3))) * mn+(mn-mx) * beta));
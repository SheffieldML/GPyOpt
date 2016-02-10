# Copyright (c) 2015, Javier Gonzalez
# Copyright (c) 2015, the GPy Authors (see GPy AUTHORS.txt)
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from .base import BOModel
import numpy as np
import GPy

class GPModel(BOModel):

    analytical_gradient_prediction = True
    
    def __init__(self, kernel=None, noise_var=None, exact_feval=False, normalize_Y=True, optimizer='bfgs', max_iters=1000, optimize_restarts=1, verbose=False):
        self.kernel = kernel
        self.noise_var = noise_var
        self.exact_feval = exact_feval
        self.normalize_Y = normalize_Y
        self.optimize_restarts = optimize_restarts
        self.optimizer = optimizer
        self.max_iters = max_iters
        self.verbose = verbose
        self.model = None
        
    def _create_model(self, X, Y):
        
        # --- define kernel
        self.input_dim = X.shape[1]
        if self.kernel is None: 
            kern = GPy.kern.RBF(self.input_dim, variance=1.) + GPy.kern.Bias(self.input_dim)
        else:
            kern = self.kernel
            self.kernel = None
        
        # --- define model
        noise_var = Y.var()*0.01 if self.noise_var is None else self.noise_var
        self.model = GPy.models.GPRegression(X, Y, kernel=kern, noise_var=noise_var)

        # --- restrict variance if exact evaluations of the objective
        if self.exact_feval:
            self.model.Gaussian_noise.constrain_fixed(1e-6, warning=False)
        else: 
            self.model.Gaussian_noise.constrain_positive(warning=False)
            
    def updateModel(self, X_all, Y_all, X_new, Y_new):
        if self.normalize_Y:
            Y_all = (Y_all - Y_all.mean())/(Y_all.std())
        if self.model is None: self._create_model(X_all, Y_all)
        else: 
            self.model.set_XY(X_all, Y_all)
            
        # --- update the model maximixing the marginal likelihood.
        if self.optimize_restarts==1:
            self.model.optimize(optimizer=self.optimizer, max_iters = self.max_iters, messages=self.verbose)
        else:
            self.model.optimize_restarts(num_restarts=self.optimize_restarts, optimizer=self.optimizer, max_iters = self.max_iters, messages=self.verbose)

    def predict(self, X):
        if X.ndim==1: X = X[None,:]
        m, v = self.model.predict(X)
        v = np.clip(v, 1e-10, np.inf)
        return m, np.sqrt(v)

    def get_fmin(self):
        return self.model.predict(self.model.X)[0].min()
    
    def predict_withGradients(self, X):
        if X.ndim==1: X = X[None,:]
        m, v = self.model.predict(X)
        v = np.clip(v, 1e-10, np.inf)
        dmdx, dvdx = self.model.predictive_gradients(X)
        dmdx = dmdx[:,:,0]
        dsdx = dvdx / (2*np.sqrt(v))
        return m, np.sqrt(v), dmdx, dsdx
    
class GPModel_MCMC(BOModel):
    
    MCMC_sampler = True
    analytical_gradient_prediction = True
    
    def __init__(self, kernel=None, noise_var=None, exact_feval=False, normalize_Y=True, n_samples = 10, n_burnin = 100, subsample_interval = 10, step_size = 1e-1, leapfrog_steps=20, verbose=False):
        self.kernel = kernel
        self.noise_var = noise_var
        self.exact_feval = exact_feval
        self.normalize_Y = normalize_Y
        self.verbose = verbose
        self.n_samples = n_samples
        self.subsample_interval = subsample_interval
        self.n_burnin = n_burnin
        self.step_size = step_size
        self.leapfrog_steps = leapfrog_steps
        self.model = None
        
    def _create_model(self, X, Y):
        
        # --- define kernel
        self.input_dim = X.shape[1]
        if self.kernel is None: 
            kern = GPy.kern.RBF(self.input_dim, variance=1.)
        else:
            kern = self.kernel
            self.kernel = None
        
        # --- define model
        noise_var = Y.var()*0.01 if self.noise_var is None else self.noise_var
        self.model = GPy.models.GPRegression(X, Y, kernel=kern, noise_var=noise_var)
        
        # --- Define priors on the hyperparameters for the kernel (for integrated acquisitions)
        self.model.kern.set_prior(GPy.priors.Gamma.from_EV(2.,4.))
        self.model.likelihood.variance.set_prior(GPy.priors.Gamma.from_EV(2.,4.))

        # --- restrict variance if exact evaluations of the objective
        if self.exact_feval:
            self.model.Gaussian_noise.constrain_fixed(1e-6, warning=False)
        else: 
            self.model.Gaussian_noise.constrain_positive(warning=False)
            
    def updateModel(self, X_all, Y_all, X_new, Y_new):
        if self.normalize_Y:
            Y_all = (Y_all - Y_all.mean())/(Y_all.std())
        if self.model is None: self._create_model(X_all, Y_all)
        else: 
            self.model.set_XY(X_all, Y_all)
            
        # update the model generating hmc samples  (?? need the first optimization?)
        self.model.optimize(max_iters = 200)
        self.model.param_array[:] = self.model.param_array * (1.+np.random.randn(self.model.param_array.size)*0.01)
        self.hmc = GPy.inference.mcmc.HMC(self.model, stepsize=self.step_size)
        ss = self.hmc.sample(num_samples=self.n_burnin + self.n_samples* self.subsample_interval, hmc_iters=self.leapfrog_steps)
        self.hmc_samples = ss[self.n_burnin::self.subsample_interval]

    def predict(self, X):
        if X.ndim==1: X = X[None,:]
        ps = self.model.param_array.copy()
        means = []
        stds = []
        for s in self.hmc_samples:
            if self.model._fixes_ is None:
                self.model[:] = s
            else:
                self.model[self.model._fixes_] = s
            self.model._trigger_params_changed()
            m, v = self.model.predict(X)
            means.append(m)
            stds.append(np.sqrt(np.clip(v, 1e-10, np.inf)))
        self.model.param_array[:] = ps
        self.model._trigger_params_changed()
        return means, stds

    def get_fmin(self):
        return self.model.predict(self.model.X)[0].min()
    
    def predict_withGradients(self, X):
        if X.ndim==1: X = X[None,:]
        ps = self.model.param_array.copy()
        means = []
        stds = []
        dmdxs = []
        dsdxs = []
        for s in self.hmc_samples:
            if self.model._fixes_ is None:
                self.model[:] = s
            else:
                self.model[self.model._fixes_] = s
            self.model._trigger_params_changed()
            m, v = self.model.predict(X)
            std = np.sqrt(np.clip(v, 1e-10, np.inf))
            dmdx, dvdx = self.model.predictive_gradients(X)
            dmdx = dmdx[:,:,0]
            dsdx = dvdx / (2*std)
            means.append(m)
            stds.append(std)
            dmdxs.append(dmdx)
            dsdxs.append(dsdx)
        self.model.param_array[:] = ps
        self.model._trigger_params_changed()
        return means, stds, dmdxs, dsdxs
    
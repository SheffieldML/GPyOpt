# Copyright (c) 2015, Javier Gonzalez
# Copyright (c) 2015, the GPy Authors (see GPy AUTHORS.txt)
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from .base import BOModel
import numpy as np
import GPy

class GPModel(BOModel):
    
    def __init__(self, kernel=None, noise_var=None, exact_feval=False, normalize_Y=True, optimizer='bfgs', max_iters=1000, optimize_restarts=1, num_hmc_samples = None, verbose=False):
        self.kernel = kernel
        self.noise_var = noise_var
        self.exact_feval = exact_feval
        self.normalize_Y = normalize_Y
        self.optimize_restarts = optimize_restarts
        self.optimizer = optimizer
        self.max_iters = max_iters
        self.verbose = verbose
        self.model = None
        self.hmc_burnin_samples = 100
        self.hmc_subsample_interval = 10
        self.num_hmc_samples = num_hmc_samples
        
    def _create_model(self, X, Y):
        import GPy
        
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
        
        # --- Define priors on the hyperparameters for the kernel (for integrated acquisitions)
        if self.num_hmc_samples != None:
            self.model.kern.set_prior(GPy.priors.Gamma.from_EV(1.,10.))

        # --- restrict variance if exact evaluations of the objective
        if self.exact_feval:
            self.model.Gaussian_noise.constrain_fixed(1e-6, warning=False)
        else: 
            self.model.Gaussian_noise.constrain_bounded(1e-6,1e6, warning=False)
            
    def updateModel(self, X_all, Y_all, X_new, Y_new):
        if self.normalize_Y:
            Y_all = (Y_all - Y_all.mean())/(Y_all.std())
        if self.model is None: self._create_model(X_all, Y_all)
        else: 
            self.model.set_XY(X_all, Y_all)
            
        if self.num_hmc_samples==None:
            # --- update the model maximixing the marginal likelihood.
            if self.optimize_restarts==1:
                self.model.optimize(optimizer=self.optimizer, max_iters = self.max_iters, messages=self.verbose)
            else:
                self.model.optimize_restarts(num_restarts=self.optimize_restarts, optimizer=self.optimizer, max_iters = self.max_iters, messages=self.verbose)
        else:
            # update the model generating hmc samples  (?? need the first optimization?)
            #self.model.optimize(optimizer=self.optimizer, max_iters = self.max_iters, messages=self.verbose)
            self.hmc = GPy.inference.mcmc.HMC(self.model,stepsize=1e-2)
            ss = self.hmc.sample(num_samples=self.hmc_burnin_samples + self.num_hmc_samples* self.hmc_subsample_interval)
            self.hmc_samples = ss[self.hmc_burnin_samples::self.hmc_subsample_interval]


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
    
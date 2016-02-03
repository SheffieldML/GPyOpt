# Copyright (c) 2015, Javier Gonzalez
# Copyright (c) 2015, the GPy Authors (see GPy AUTHORS.txt)
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from .base import BOModel
import numpy as np

class GPModel(BOModel):
    
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
        import GPy
        
        self.input_dim = X.shape[1]
        if self.kernel is None: 
            kern = GPy.kern.RBF(self.input_dim, variance=1.)+GPy.kern.Bias(self.input_dim)
        else:
            kern = self.kernel
            self.kernel = None
            
        noise_var = Y.var()*0.01 if self.noise_var is None else self.noise_var

        self.model = GPy.models.GPRegression(X, Y, kernel=kern, noise_var=noise_var)

        if self.exact_feval:
            self.model.Gaussian_noise.constrain_fixed(1e-6, warning=False)
            
    def updateModel(self, X_all, Y_all, X_new, Y_new):
        if self.normalize_Y:
            Y_all = (Y_all - Y_all.mean())/(Y_all.std())
        if self.model is None: self._create_model(X_all, Y_all)
        else: 
            self.model.set_XY(X_all, Y_all)
            
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
    
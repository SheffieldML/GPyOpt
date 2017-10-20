# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
import GPy

from .base import BOModel


##
## TODO: not fully tested yet.
##


class WarpedGPModel(BOModel):

    analytical_gradient_prediction = False

    def __init__(self, kernel=None, noise_var=None, exact_feval=False, optimizer='bfgs', max_iters=1000,
                        optimize_restarts=5, warping_function=None, warping_terms=3, verbose=False):

        self.kernel = kernel
        self.noise_var = noise_var
        self.exact_feval = exact_feval
        self.optimize_restarts = optimize_restarts
        self.optimizer = optimizer
        self.max_iters = max_iters
        self.verbose = verbose
        self.warping_function = warping_function
        self.warping_terms =  warping_terms
        self.model = None

    def _create_model(self, X, Y):
        # --- define kernel
        self.input_dim = X.shape[1]
        if self.kernel is None:
            self.kernel = GPy.kern.Matern32(self.input_dim, variance=1.) #+ GPy.kern.Bias(self.input_dim)
        else:
            self.kernel = self.kernel

        # --- define model
        noise_var = Y.var()*0.01 if self.noise_var is None else self.noise_var

        self.model = GPy.models.WarpedGP(X, Y, kernel=self.kernel, warping_function=self.warping_function, warping_terms=self.warping_terms )

        # --- restrict variance if exact evaluations of the objective
        if self.exact_feval:
            self.model.Gaussian_noise.constrain_fixed(1e-6, warning=False)
        else:
                self.model.Gaussian_noise.constrain_positive(warning=False)

    def updateModel(self, X_all, Y_all, X_new, Y_new):
        if self.model is None:
            self._create_model(X_all, Y_all)
        else:
            self.model.set_XY(X_all, Y_all)

        self.model.optimize(optimizer = self.optimizer, messages=self.verbose, max_iters=self.max_iters)


    def predict(self, X):
        if X.ndim==1: X = X[None,:]
        m, v = self.model.predict(X)
        v = np.clip(v, 1e-10, np.inf)
        return m, np.sqrt(v)

    def get_fmin(self):
        return self.model.predict(self.model.X)[0].min()
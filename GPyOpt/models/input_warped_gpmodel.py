# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from .gpmodel import GPModel
import numpy as np
import GPy


class InputWarpedGPModel(GPModel):
    """Bayesian Optimization with Input Warped GP using Kumar Warping

    The Kumar warping only applies to the numerical variables: continuous and discrete

    Parameters
    ----------
    space : object
        Instance of Design_space defined in GPyOpt.core.task.space

    warping_function : object, optional
        Warping function defined in GPy.util.input_warping_functions.py. Default is Kumar warping

    kernel : object, optional
        An instance of kernel function defined in GPy.kern. Default is Matern 52

    noise_var : float, optional
        Value of the noise variance if known

    exact_feval : bool, optional
        Whether noiseless evaluations are available.
        IMPORTANT to make the optimization work well in noiseless scenarios, Default is False

    optimizer : string, optional
        Optimizer of the model. Check GPy for details. Default to bfgs

    max_iter : int, optional
        Maximum number of iterations used to optimize the parameters of the model. Default is 1000

    optimize_restarts : int, optional
        Number of restarts in the optimization. Default is 5

    verbose : bool, optional
        Whether to print out the model messages. Default is False
    """

    analytical_gradient_prediction = False

    def __init__(self, space, warping_function=None, kernel=None, noise_var=None, exact_feval=False, optimizer='bfgs',
                 max_iters=1000, optimize_restarts=5, verbose=False, ARD=False):
        self.space = space
        # set the warping indices
        self.warping_indices = []
        i = 0
        for var in self.space.space:
            for _ in range(var.dimensionality):
                if var.type == 'continuous' or var.type == 'discrete':
                    self.warping_indices.append(i)
                i += 1
        self.warping_function = warping_function

        self.kernel = kernel
        self.noise_var = noise_var
        self.exact_feval = exact_feval
        self.optimize_restarts = optimize_restarts
        self.optimizer = optimizer
        self.max_iters = max_iters
        self.verbose = verbose
        self.model = None
        self.ARD = ARD

    def _create_model(self, X, Y):
        # --- define kernel
        self.input_dim = X.shape[1]
        if self.kernel is None:
            self.kernel = GPy.kern.Matern52(self.input_dim, variance=1., ARD=self.ARD) #+ GPy.kern.Bias(self.input_dim)
        else:
            self.kernel = self.kernel

        # --- define model
        noise_var = Y.var()*0.01 if self.noise_var is None else self.noise_var

        self.model = GPy.models.InputWarpedGP(X, Y, kernel=self.kernel, warping_function=self.warping_function,
                                              warping_indices=self.warping_indices, Xmin=X.min(axis=0), Xmax=X.max(axis=0))

        # --- restrict variance if exact evaluations of the objective
        if self.exact_feval:
            self.model.Gaussian_noise.constrain_fixed(1e-6, warning=False)
        else:
            self.model.Gaussian_noise.constrain_bounded(1e-9, 1e6, warning=False)
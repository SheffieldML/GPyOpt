# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
import GPy

from .base import BOModel

class GPModel(BOModel):
    """
    General class for handling a Gaussian Process in GPyOpt.

    :param kernel: GPy kernel to use in the GP model.
    :param noise_var: value of the noise variance if known.
    :param exact_feval: whether noiseless evaluations are available. IMPORTANT to make the optimization work well in noiseless scenarios (default, False).
    :param optimizer: optimizer of the model. Check GPy for details.
    :param max_iters: maximum number of iterations used to optimize the parameters of the model.
    :param optimize_restarts: number of restarts in the optimization.
    :param sparse: whether to use a sparse GP (default, False). This is useful when many observations are available.
    :param num_inducing: number of inducing points if a sparse GP is used.
    :param verbose: print out the model messages (default, False).
    :param ARD: whether ARD is used in the kernel (default, False).

    .. Note:: This model does Maximum likelihood estimation of the hyper-parameters.

    """


    analytical_gradient_prediction = True  # --- Needed in all models to check is the gradients of acquisitions are computable.

    def __init__(self, kernel=None, noise_var=None, exact_feval=False, optimizer='bfgs', max_iters=1000, optimize_restarts=5, sparse = False, num_inducing = 10,  verbose=True, ARD=False):
        self.kernel = kernel
        self.noise_var = noise_var
        self.exact_feval = exact_feval
        self.optimize_restarts = optimize_restarts
        self.optimizer = optimizer
        self.max_iters = max_iters
        self.verbose = verbose
        self.sparse = sparse
        self.num_inducing = num_inducing
        self.model = None
        self.ARD = ARD

    @staticmethod
    def fromConfig(config):
        return GPModel(**config)

    def _create_model(self, X, Y):
        """
        Creates the model given some input data X and Y.
        """

        # --- define kernel
        self.input_dim = X.shape[1]
        if self.kernel is None:
            kern = GPy.kern.Matern52(self.input_dim, variance=1., ARD=self.ARD) #+ GPy.kern.Bias(self.input_dim)
        else:
            kern = self.kernel
            self.kernel = None

        # --- define model
        noise_var = Y.var()*0.01 if self.noise_var is None else self.noise_var

        if not self.sparse:
            self.model = GPy.models.GPRegression(X, Y, kernel=kern, noise_var=noise_var)
        else:
            self.model = GPy.models.SparseGPRegression(X, Y, kernel=kern, num_inducing=self.num_inducing)

        # --- restrict variance if exact evaluations of the objective
        if self.exact_feval:
            self.model.Gaussian_noise.constrain_fixed(1e-6, warning=False)
        else:
            # --- We make sure we do not get ridiculously small residual noise variance
            self.model.Gaussian_noise.constrain_bounded(1e-9, 1e6, warning=False) #constrain_positive(warning=False)

    def updateModel(self, X_all, Y_all, X_new, Y_new):
        """
        Updates the model with new observations.
        """
        if self.model is None:
            self._create_model(X_all, Y_all)
        else:
            self.model.set_XY(X_all, Y_all)

        # WARNING: Even if self.max_iters=0, the hyperparameters are bit modified...
        if self.max_iters > 0:
            # --- update the model maximizing the marginal likelihood.
            if self.optimize_restarts==1:
                self.model.optimize(optimizer=self.optimizer, max_iters = self.max_iters, messages=False, ipython_notebook=False)
            else:
                self.model.optimize_restarts(num_restarts=self.optimize_restarts, optimizer=self.optimizer, max_iters = self.max_iters, verbose=self.verbose)

    def _predict(self, X, full_cov, include_likelihood):
        if X.ndim == 1:
            X = X[None,:]
        m, v = self.model.predict(X, full_cov=full_cov, include_likelihood=include_likelihood)
        v = np.clip(v, 1e-10, np.inf)
        return m, v

    def predict(self, X, with_noise=True):
        """
        Predictions with the model. Returns posterior means and standard deviations at X. Note that this is different in GPy where the variances are given.

        Parameters:
            X (np.ndarray) - points to run the prediction for.
            with_noise (bool) - whether to add noise to the prediction. Default is True.
        """
        m, v = self._predict(X, False, with_noise)
        # We can take the square root because v is just a diagonal matrix of variances
        return m, np.sqrt(v)

    def predict_covariance(self, X, with_noise=True):
        """
        Predicts the covariance matric for points in X.

        Parameters:
            X (np.ndarray) - points to run the prediction for.
            with_noise (bool) - whether to add noise to the prediction. Default is True.
        """
        _, v = self._predict(X, True, with_noise)
        return v

    def get_fmin(self):
        """
        Returns the location where the posterior mean is takes its minimal value.
        """
        return self.model.predict(self.model.X)[0].min()

    def predict_withGradients(self, X):
        """
        Returns the mean, standard deviation, mean gradient and standard deviation gradient at X.
        """
        if X.ndim==1: X = X[None,:]
        m, v = self.model.predict(X)
        v = np.clip(v, 1e-10, np.inf)
        dmdx, dvdx = self.model.predictive_gradients(X)
        dmdx = dmdx[:,:,0]
        dsdx = dvdx / (2*np.sqrt(v))

        return m, np.sqrt(v), dmdx, dsdx

    def copy(self):
        """
        Makes a safe copy of the model.
        """
        copied_model = GPModel(kernel = self.model.kern.copy(),
                            noise_var=self.noise_var,
                            exact_feval=self.exact_feval,
                            optimizer=self.optimizer,
                            max_iters=self.max_iters,
                            optimize_restarts=self.optimize_restarts,
                            verbose=self.verbose,
                            ARD=self.ARD)

        copied_model._create_model(self.model.X,self.model.Y)
        copied_model.updateModel(self.model.X,self.model.Y, None, None)
        return copied_model

    def get_model_parameters(self):
        """
        Returns a 2D numpy array with the parameters of the model
        """
        return np.atleast_2d(self.model[:])

    def get_model_parameters_names(self):
        """
        Returns a list with the names of the parameters of the model
        """
        return self.model.parameter_names_flat().tolist()

    def get_covariance_between_points(self, x1, x2):
        """
        Given the current posterior, computes the covariance between two sets of points.
        """
        return self.model.posterior_covariance_between_points(x1, x2)


class GPModel_MCMC(BOModel):
    """
    General class for handling a Gaussian Process in GPyOpt.

    :param kernel: GPy kernel to use in the GP model.
    :param noise_var: value of the noise variance if known.
    :param exact_feval: whether noiseless evaluations are available. IMPORTANT to make the optimization work well in noiseless scenarios (default, False).
    :param n_samples: number of MCMC samples.
    :param n_burnin: number of samples not used.
    :param subsample_interval: sub-sample interval in the MCMC.
    :param step_size: step-size in the MCMC.
    :param leapfrog_steps: ??
    :param verbose: print out the model messages (default, False).

    .. Note:: This model does MCMC over the hyperparameters.

    """

    MCMC_sampler = True
    analytical_gradient_prediction = True # --- Needed in all models to check is the gradients of acquisitions are computable.

    def __init__(self, kernel=None, noise_var=None, exact_feval=False, n_samples = 10, n_burnin = 100, subsample_interval = 10, step_size = 1e-1, leapfrog_steps=20, verbose=False):
        self.kernel = kernel
        self.noise_var = noise_var
        self.exact_feval = exact_feval
        self.verbose = verbose
        self.n_samples = n_samples
        self.subsample_interval = subsample_interval
        self.n_burnin = n_burnin
        self.step_size = step_size
        self.leapfrog_steps = leapfrog_steps
        self.model = None

    def _create_model(self, X, Y):
        """
        Creates the model given some input data X and Y.
        """

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

        # --- Define prior on the hyper-parameters for the kernel (for integrated acquisitions)
        self.model.kern.set_prior(GPy.priors.Gamma.from_EV(2.,4.))
        self.model.likelihood.variance.set_prior(GPy.priors.Gamma.from_EV(2.,4.))

        # --- Restrict variance if exact evaluations of the objective
        if self.exact_feval:
            self.model.Gaussian_noise.constrain_fixed(1e-6, warning=False)
        else:
            self.model.Gaussian_noise.constrain_positive(warning=False)

    def updateModel(self, X_all, Y_all, X_new, Y_new):
        """
        Updates the model with new observations.
        """

        if self.model is None:
            self._create_model(X_all, Y_all)
        else:
            self.model.set_XY(X_all, Y_all)

        # update the model generating hmc samples
        self.model.optimize(max_iters = 200)
        self.model.param_array[:] = self.model.param_array * (1.+np.random.randn(self.model.param_array.size)*0.01)
        self.hmc = GPy.inference.mcmc.HMC(self.model, stepsize=self.step_size)
        ss = self.hmc.sample(num_samples=self.n_burnin + self.n_samples* self.subsample_interval, hmc_iters=self.leapfrog_steps)
        self.hmc_samples = ss[self.n_burnin::self.subsample_interval]

    def predict(self, X):
        """
        Predictions with the model for all the MCMC samples. Returns posterior means and standard deviations at X. Note that this is different in GPy where the variances are given.
        """

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
        """
        Returns the location where the posterior mean is takes its minimal value.
        """
        ps = self.model.param_array.copy()
        fmins = []
        for s in self.hmc_samples:
            if self.model._fixes_ is None:
                self.model[:] = s
            else:
                self.model[self.model._fixes_] = s
            self.model._trigger_params_changed()
            fmins.append(self.model.predict(self.model.X)[0].min())
        self.model.param_array[:] = ps
        self.model._trigger_params_changed()

        return fmins

    def predict_withGradients(self, X):
        """
        Returns the mean, standard deviation, mean gradient and standard deviation gradient at X for all the MCMC samples.
        """
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

    def copy(self):
        """
        Makes a safe copy of the model.
        """

        copied_model = GPModel( kernel = self.model.kern.copy(),
                                noise_var= self.noise_var ,
                                exact_feval= self.exact_feval,
                                n_samples = self.n_samples,
                                n_burnin = self.n_burnin,
                                subsample_interval = self.subsample_interval,
                                step_size = self.step_size,
                                leapfrog_steps= self.leapfrog_steps,
                                verbose= self.verbose)

        copied_model._create_model(self.model.X,self.model.Y)
        copied_model.updateModel(self.model.X,self.model.Y, None, None)
        return copied_model

    def get_model_parameters(self):
        """
        Returns a 2D numpy array with the parameters of the model
        """
        return np.atleast_2d(self.model[:])

    def get_model_parameters_names(self):
        """
        Returns a list with the names of the parameters of the model
        """
        return self.model.parameter_names()

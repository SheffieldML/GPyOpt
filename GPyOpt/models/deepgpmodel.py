# Copyright (c) 2015, Javier Gonzalez
# Copyright (c) 2015, the GPy Authors (see GPy AUTHORS.txt)
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from .base import BOModel
import numpy as np
import GPy
import deepgp
from GPy.core.parameterization.variational import VariationalPosterior, NormalPosterior


class DeepGPModel(BOModel):

    analytical_gradient_prediction = False
    
    def __init__(self, kernel=None, noise_var=None, exact_feval=False, normalize_Y=True, optimizer='bfgs', max_iters=1000, 
    			optimize_restarts=5, num_inducing = 10, back_constraint=True, repeatX=True, verbose=False, max_init_iters=100):          

        self.noise_var = noise_var
        self.exact_feval = exact_feval
        self.normalize_Y = normalize_Y
        self.optimize_restarts = optimize_restarts
        self.optimizer = optimizer
        self.max_iters = max_iters
        self.max_init_iters = max_init_iters
        self.verbose = verbose
        self.back_constraint =back_constraint
        self.repeatX =repeatX
        self.model_num_inducing = num_inducing
        self.model_kernel = kernel
        self.Ds = 1


    def _create_model(self, X, Y):
        '''
        Initializes a deep Gaussian Process with one hidden layer over *f*.
        :param X: input observations.
        :param Y: output values.
        '''

        self.X = X
        self.Y = Y

        self.useGPU = False

        # --- kernel and dimension of the hidden layer
        if self.model_kernel == None:
        	# self.kernel = [GPy.kern.Matern32(self.Ds, ARD=False), GPy.kern.Matern32(self.X.shape[1], ARD=False)]
            self.kernel = [GPy.kern.RBF(self.Ds, ARD=True), GPy.kern.RBF(self.X.shape[1], ARD=True)]
        else:
        	self.kernel = [k.copy() for k in self.model_kernel]  # this need to be one kernel per layer

        # Type of deepGPs
        if self.back_constraint:
            self.model = deepgp.DeepGP([self.Y.shape[1],self.Ds, self.X.shape[1]], Y=self.Y, X=self.X, num_inducing=self.num_inducing, kernels=self.kernel, MLP_dims=[[100,50],[]], repeatX=self.repeatX)
        else:
            self.model = deepgp.DeepGP([self.Y.shape[1],self.Ds, self.X.shape[1]], Y=self.Y, X=self.X, num_inducing=self.num_inducing, kernels=self.kernel, back_constraint=False, repeatX=self.repeatX)

        if self.exact_feval == True:
            self.model.obslayer.Gaussian_noise.constrain_fixed(1e-6, warning=False) #to avoid numerical problems
        else:
            self.model.obslayer.Gaussian_noise.constrain_positive(warning=False) #to avoid numerical problems


    def updateModel(self, X_all, Y_all, X_new, Y_new):
        import numpy as np

        if self.normalize_Y:
        	Y_all = (Y_all - Y_all.mean())/(Y_all.std())

        # Do not use more inducing than data
        self.num_inducing = np.min((self.model_num_inducing, Y_all.shape[0]))
        self._create_model(X_all, Y_all)  # we re-create the model because set_XY is not available.

		# Model optimization
        for i in range(len(self.model.layers)):
            if isinstance(self.model.layers[i].Y, NormalPosterior) or isinstance(self.model.layers[i].Y, VariationalPosterior):
                cur_var = self.model.layers[i].Y.mean.var()
            else:
                cur_var = self.model.layers[i].Y.var()
            self.model.layers[i].Gaussian_noise.variance = cur_var / 100.

            self.model.layers[i].Gaussian_noise.variance.fix(warning=False)

        self.model.optimize(optimizer = self.optimizer, messages=self.verbose, max_iters=self.max_init_iters)

        for i in range(len(self.model.layers)):
            self.model.layers[i].Gaussian_noise.variance.constrain_positive(warning=False)

        self.model.optimize(optimizer = self.optimizer, messages=self.verbose, max_iters=self.max_iters)
        #deepgp.util.check_snr(self.model) 



    def predict(self, X):
        if X.ndim==1: X = X[None,:]
        m, v = self.model.predict(X)
        v = np.clip(v, 1e-10, np.inf)
        return m, np.sqrt(v)

    def get_fmin(self):
    	return self.model.predict(self.model.X)[0].min()

    ## TODO: no predictive gradients so far in the deepGP models    
    # def predict_withGradients(self, X):


    

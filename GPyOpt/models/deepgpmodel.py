# Copyright (c) 2015, Javier Gonzalez
# Copyright (c) 2015, the GPy Authors (see GPy AUTHORS.txt)
# Licensed under the BSD 3-clause license (see LICENSE.txt)

## TODO

from .base import BOModel
import numpy as np
import GPy
import deepgp


class DeepGPModel(BOModel):

    analytical_gradient_prediction = False
    

    def __init__(self, kernel=None, noise_var=None, exact_feval=False, normalize_Y=True, optimizer='bfgs', max_iters=1000, 
    			optimize_restarts=5, num_inducing = 10, back_constraint=True, repeatX=True, Ds = 1, verbose=False):          

        self.noise_var = noise_var
        self.exact_feval = exact_feval
        self.normalize_Y = normalize_Y
        self.optimize_restarts = optimize_restarts
        self.optimizer = optimizer
        self.max_iters = max_iters
        self.verbose = verbose
        self.back_constraint =back_constraint
        self.repeatX =repeatX
        self.num_inducing = num_inducing
        self.kernel = kernel
        self.Ds = Ds



    def _create_model(self, X, Y):
        '''
        Initializes a deep Gaussian Process with one hidden layer over *f*.
        :param X: input observations.
        :param Y: output values.
        '''

        self.X = X
        self.Y = Y

        import socket
        self.useGPU = False
        if socket.gethostname()[0:4] == 'node':
            print 'Using GPU!'
            self.useGPU = True

        # --- kernel and dimension of the hidden layer
        if self.kernel == None:
        	self.kernel = [GPy.kern.Matern32(self.Ds, ARD=False), GPy.kern.Matern32(self.X.shape[1], ARD=False)]
        else:
        	self.kernel = self.kernel  # this need to be one kernel per layer

        # Type of deepGPs
        if self.back_constraint:
            self.model = deepgp.DeepGP([self.Y.shape[1],self.Ds, self.X.shape[1]], Y=self.Y, X=self.X, num_inducing=self.num_inducing, kernels=self.kernel, MLP_dims=[[100,50],[]]) #, repeatX=self.repeatX)
        
        else:
            self.model = deepgp.DeepGP([self.Y.shape[1],self.Ds, self.X.shape[1]], Y=self.Y, X=self.X, num_inducing=self.num_inducing, kernels=self.kernel, back_constraint=False) #, repeatX=self.repeatX)

        if self.exact_feval == True:
            self.model.obslayer.Gaussian_noise.constrain_fixed(1e-6, warning=False) #to avoid numerical problems
        else:
            self.model.obslayer.Gaussian_noise.constrain_bounded(1e-6,1e6, warning=False) #to avoid numerical problems


    def updateModel(self, X_all, Y_all, X_new, Y_new):

        if self.normalize_Y:
        	Y_all = (Y_all - Y_all.mean())/(Y_all.std())

		self._create_model(X_all, Y_all)

		# Optimization
		# self.m_init = self.model.copy()
		# self.model.obslayer['Gaussian_noise.variance'].fix()
		# self.model.layer_1['Gaussian_noise.variance'].fix()
		# self.model.optimize(optimizer = self.optimizer, messages=self.verbose, max_iters=self.max_iters)
		# self.model_init2 = self.modelcopy()
		# self.model.obslayer['Gaussian_noise.variance'].constrain_positive()
		# self.model.layer_1['Gaussian_noise.variance'].constrain_positive()
		self.model.optimize(optimizer = self.optimizer, messages=self.verbose, max_iters=self.max_iters)


    def predict(self, X):
        if X.ndim==1: X = X[None,:]
        m, v = self.model.predict(X)
        v = np.clip(v, 1e-10, np.inf)
        return m, np.sqrt(v)

    def get_fmin(self):
    	return self.model.predict(self.model.X)[0].min()
    
    # def predict_withGradients(self, X):
    #     if X.ndim==1: X = X[None,:]
    #     m, v = self.model.predict(X)
    #     v = np.clip(v, 1e-10, np.inf)
    #     dmdx, dvdx = self.model.predictive_gradients(X)
    #     dmdx = dmdx[:,:,0]
    #     dsdx = dvdx / (2*np.sqrt(v))
    #     return m, np.sqrt(v), dmdx, dsdx

    

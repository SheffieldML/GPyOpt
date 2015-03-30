# Copyright (c) 2015, Javier Gonzalez
# Copyright (c) 2015, the GPy Authors (see GPy AUTHORS.txt)
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import GPy

from ..core.acquisition import AcquisitionEI, AcquisitionMPI, AcquisitionLCB 
from ..core.bo import BO
from ..util.general import samples_multidimensional_uniform


class BayesianOptimization(BO):
    def __init__(self, f, bounds=None, kernel=None, X=None, Y=None, optimize_model=None, model_optimize_interval=1, model_optimize_restarts=5, acquisition='EI', acquisition_par= 0.01,  model_data_init = None, sparse=False, num_inducing=None, normalize=False, verbosity=0):
        '''
        Bayesian Optimization using EI, MPI and LCB (or UCB) acquisition functions.
    
        This is a thin wrapper around the methods.BO class, with a set of sensible defaults
        :param f the function to optimize.
        :param bounds: Tuple containing the box constrains of the function to optimize. Example: for [0,1]x[0,1] insert [(0,1),(0,1)].  
        :param X: input observations
        :param Y: output values
        :param kernel: a GPy kernel, defaults to rbf + bias.
        :param optimize_model: Unless specified otherwise the parameters of the model are updated after each iteration. 
        :param model_optimize_interval: number of iterations after which the parameters of the model are optimized.   
        :param model_optimize_restarts: number of initial points for the GP parameters optimization (5, default)
        :param acquisition: acquisition function ('EI' 'MPI' or LCB). Default, EI.
        :param acquisition_par: parameter of the acquisition function. 
        :param model_data_init: number of initial random evaluations of f is X and Y are not provided (default, 3*input_dim).  
        :param sparse: whether to use an sparse GP (False, default).
        :param num_inducing: number of inducing points for a Sparse GP (None, default)
        :param normalize: whether to normalize the Y's for optimization (False, default).
        :param true_gradients: whether the true gradients of the acquisition function are used for optimization (True, default). 
        :param verbosity: whether to show (1) or not (0, default) the value of the log-likelihood of the model for the optimized parameters.
    
        '''
        self.model_data_init = model_data_init  
        self.num_inducing = num_inducing
        self.sparse = sparse
        self.input_dim = len(bounds)
        self.normalize = normalize
        if f==None: 
            print 'Function to optimize is required.'
        else:
            self.f = f  

        # ------- Initialize model 
        if bounds==None: 
            raise 'Box constraints are needed. Please insert box constrains.' 
        else:
            self.bounds = bounds
        if  model_data_init ==None:
            self.model_data_init = 3*self.input_dim
        else:
            self.model_data_init = model_data_init
        if X==None or Y == None:
            self.X = samples_multidimensional_uniform(self.bounds, self.model_data_init)
            self.Y = f(self.X)
        else:
            self.X = X
            self.Y = Y
        if kernel is None: 
            self.kernel = GPy.kern.RBF(self.input_dim, variance=.1, lengthscale=.1) + GPy.kern.Bias(self.input_dim)
        else:
            self.kernel = kernel
        self._init_model()
        

        # ------- Initialize acquisition function
        self.acqu_name = acquisition
        if  acquisition_par == None:
            self.acquisition_par = 0
        else:
            self.acquisition_par = acquisition_par
        
        if acquisition==None or acquisition=='EI': 
            acq = AcquisitionEI(acquisition_par)
        elif acquisition=='MPI':
            acq = AcquisitionMPI(acquisition_par)
        elif acquisition=='LCB':
            acq = AcquisitionLCB(acquisition_par)
        else:   
            print 'The selected acquisition function is not valid. Please try again with EI, MPI, or LCB'
        if (acquisition=='EI' or acquisition=='MPI' or acquisition =='LCB'):
            super(BayesianOptimization ,self).__init__(acquisition_func=acq, bounds=bounds, model_optimize_interval=model_optimize_interval, model_optimize_restarts=model_optimize_restarts, model_data_init=model_data_init, normalize=normalize, verbosity=verbosity)
    
    
    def _init_model(self):
        '''
        Initializes the Gaussian Process over *f*.
        :param X: input observations.
        :param Y: output values.

        ..Note : X and Y can be None. In this case Nrandom*model_dimension data are uniformly generated to initialize the model.
        
        '''
        if self.sparse == True:
            if self.num_inducing ==None:
                raise 'Sparse model, please insert the number of inducing points'
            else:           
                self.model = GPy.models.SparseGPRegression(self.X, self.Y, kernel=self.kernel, num_inducing=self.num_inducing)
        else:       
            self.model = GPy.models.GPRegression(self.X,self.Y,kernel=self.kernel)


# Copyright (c) 2014, Javier Gonzalez
# Copyright (c) 2014, the GPy Authors (see GPy AUTHORS.txt)
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import GPy
from ..core.acquisition import AcquisitionEI, AcquisitionMPI, AcquisitionLCB 
from ..core.bo import BO
from ..util.general import samples_multidimensional_uniform


class BayesianOptimization(BO):
    """
    Bayesian Optimization using EI, MPI and LCB (or UCB) acquisition functions.

    This is a thin wrapper around the methods.BO class, with a set of sensible defaults

    :param: f the function to optimize
    :param bounds: Tuple containing the box contrains of the function to optimize. Example: for [0,1]x[0,1] insert [(0,1),(0,1)].  
    :param X: input observations
    :param Y: output values
    :param kernel: a GPy kernel, defaults to rbf + bias.
    :param optimize_model: Unless specified otherwise the parameters of the model are updated after each iteration. 
    :param model_optimize_interval: iterations after which the parameters of the model are optimized.   
    :param model_optimize_restarts: number of initial points for the GP parameters optimization.
    :param acquisition: acquisition function ('EI' 'MPI' or LCB). Default set to EI.
    :param acquisition_par: parameter of the acquisition function. To avoid local minima.
    :param nodel_data_init: number of initial random evaluatios of f is X and Y are not provided (2*input_dim is used by default).  
    :param sparse: if sparse is True, and sparse GP is used.
    :param normalize: normalization of the Y's. Default is False.
    :param verbosity: whether to show (1) or not (0, default) the value of the log-likelihood of the model for the optimized parameters.

    .. Note:: Multiple independent outputs are allowed using columns of Y

    """
    def __init__(self, f, bounds=None, kernel=None, X=None, Y=None, optimize_model=None, model_optimize_interval=1, model_optimize_restarts=5, acquisition='EI', acquisition_par=None,  model_data_init = None, sparse=False, num_inducing=None, normalize=False, verbosity=0):
        self.model_data_init = model_data_init  
        self.num_inducing = num_inducing
        self.sparse = sparse
        self.input_dim = len(bounds)
        self.normalize = normalize
        if f==None: 
            print 'Function to optimize is requiered'
        else:
            self.f = f
        
        ## Initilize model 
        if bounds==None: 
            raise 'Box contrainst are needed. Please insert box constrains' 
        else:
            self.bounds = bounds
        if  model_data_init ==None:
            self.model_data_init = 2*self.input_dim
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
        self.acqu_name = acquisition
        if  acquisition_par == None:
            self.acquisition_par = 0
        else:
            self.acquisition_par = acquisition_par

        # Initilize aquisition function
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
        Initializes the prior measure, or Gaussian Process, over the function f to optimize

        :param X: input observations
        :param Y: output values

        ..Note : X and Y can be None. In this case Nrandom*model_dimension data are uniformly generated to initialize the model.

        '''
        if self.sparse == True:
            if self.num_inducing ==None:
                raise 'Sparse model, please insert the number of inducing points'
            else:           
                self.model = GPy.models.SparseGPRegression(self.X, self.Y, kernel=self.kernel, num_inducing=self.num_inducing)
        else:       
            self.model = GPy.models.GPRegression(self.X,self.Y,kernel=self.kernel)





# Copyright (c) 2015, Javier Gonzalez
# Copyright (c) 2015, the GPy Authors (see GPy AUTHORS.txt)
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import GPy
import deepgp
import numpy as np
from ..core.acquisition import AcquisitionEI, AcquisitionMPI, AcquisitionLCB, AcquisitionEL
from ..core.bo import BO
from ..util.general import samples_multidimensional_uniform, reshape
from ..util.stats import initial_design
import warnings
warnings.filterwarnings("ignore")


class BayesianOptimization(BO):
    def __init__(self, f, bounds=None, model_type=None, kernel=None, X=None, Y=None, numdata_inital_design = None,type_initial_design='random', model_optimize_interval=1, acquisition='EI', 
        acquisition_par= 0.00, model_optimize_restarts=10, sparseGP=False, num_inducing=None, normalize=False, 
        exact_feval=False, verbosity=0):
        '''
        Bayesian Optimization using EI, MPI and LCB (or UCB) acquisition functions.

        This is a thin wrapper around the methods.BO class, with a set of sensible defaults
        :param *f* the function to optimize. Should get a nxp numpy array as imput and return a nx1 numpy array.
        :param model: model used for the optimization: it can be 
            - 'GP Regression': used by default 'gp'
            - 'Sparse GP Regression': 'sparsegp'
            - 'Deep GP Regression': 'deepgp'
            - 'Deep GP Regression with back constraint': 'deepgp_back_constraint'
        :param bounds: Tuple containing the box constrains of the function to optimize. Example: for [0,1]x[0,1] insert [(0,1),(0,1)].
        :param kernel: a GPy kernel, defaults to rbf.
        :param X: input observations. If X=None, some  points are evaluated randomly.
        :param Y: output values. If Y=None, f(X) is used.
        :param numdata_initial_design: number of initial random evaluations of f is X and Y are not provided (default, 3*input_dim).
        :param type_initial_design: type of initial design for the X matrix:
            - 'random': random (uniform) design.
            - 'latin': latin hypercube (requieres pyDOE).
        :param model_optimize_interval: number of iterations after which the parameters of the model are optimized (1, Default).
        :param acquisition: acquisition function ('EI': Expec. Improvement. 'MPI': Maximum Prob. Improvement. 'EL': Expected Loss. LCB: Lower Confidence Bound). Default, EI.
        :param acquisition_par: parameter of the acquisition function.
        :param model_optimize_restarts: number of initial points for the GP parameters optimization (5, default)
        :param sparseGP: whether to use an sparse GP (False, default).
        :param num_inducing: number of inducing points for a Sparse GP (None, default)
        :param normalize: whether to normalize the Y's for optimization (False, default).
        :param exact_feval: set the noise variance of the GP if True (False, default).
        :param verbosity: whether to show (1) or not (0, default) the value of the log-likelihood of the model for the optimized parameters.

        '''
        # ------- Get default values 
        if num_inducing == None:    
            self.num_inducing = 30
        else:
            self.num_inducing = num_inducing
        self.sparseGP = sparseGP
        self.input_dim = len(bounds)
        self.normalize = normalize
        self.exact_feval = exact_feval
        self.model_optimize_interval = model_optimize_interval
        self.model_optimize_restarts = model_optimize_restarts
        self.verbosity = verbosity
        self.type_initial_design = type_initial_design

        if model_type == None:
            self.model_type = 'gp'
        else: 
            self.model_type = model_type
            
        if f==None: 
            print 'Function to optimize is required.'
        else:
            self.f = f

        # ------- Initialize model
        if bounds==None:
            raise 'Box constraints are needed. Please insert box constrains.'
        else:
            self.bounds = bounds
        if  numdata_initial_design==None:
            self.numdata_initial_design = 3*self.input_dim
        else:
            self.numdata_initial_design = numdata_initial_design

        # A couple cases might arise when handling initial observations:
        # (0) neither X nor Y given, (1) X but not Y given, (2) Y but not X given,
        # (3) X and Y given.
        # In case 3, display a warning and proceed as in case (0).

        # if X not given, use randomized initial design (case 0 or 2)
        if X==None:
            if Y!=None:
                warnings.warn("User supplied initial Y without matching X")
            self.X = initial_design(self.type_initial_design,self.bounds, self.numdata_initial_design)
            self.Y = f(self.X)

        # case 1: X but not Y given
        elif Y==None:
            self.X = X
            self.Y = f(self.X)

        # case 3: X and Y given
        else:
            self.X = X
            self.Y = Y

        if kernel is None:
            self.kernel = GPy.kern.RBF(self.input_dim, variance=1., lengthscale=np.max(self.bounds)/5.)+GPy.kern.Bias(self.input_dim)
        else:
            self.kernel = kernel
        self._init_model()
        self.first_time_optimization = True  # control over the optimization of the GP


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
        elif acquisition=='EL':
            acq = AcquisitionEL(acquisition_par)
        else:
            print 'The selected acquisition function is not valid. Please try again with EI, MPI, or LCB'
        if (acquisition=='EI' or acquisition=='MPI' or acquisition =='LCB' or acquisition =='EL' ):
            super(BayesianOptimization ,self).__init__(acquisition_func=acq)


    def _init_model(self):
        '''
        Initializes the Gaussian Process over *f*.
        :param X: input observations.
        :param Y: output values.

        ..Note : X and Y can be None. In this case numdata_initial_design*input_dim data are uniformly generated to initialize the model.

        '''
        # --- the model is a sparse GP

        if self.model_type == 'sparsegp' or self.model_type == 'gp':
            self._init_gp()

        elif self.model_type == 'deepgp' or  self.model_type == 'deepgp_back_constraint':
            self._init_deepgp()

            
    
    def _init_gp(self):

        if self.model_type == 'gp':
            self.model = GPy.models.GPRegression(self.X, self.Y, kernel=self.kernel)

        elif self.model_type == 'sparsegp':
            if self.num_inducing ==None:
                raise 'Sparse model, please insert the number of inducing points'
            else:
                self.model = GPy.models.SparseGPRegression(self.X, self.Y, kernel=self.kernel, num_inducing=self.num_inducing)

        if self.exact_feval == True:
            self.model.Gaussian_noise.constrain_fixed(1e-6, warning=False) #to avoid numerical problems
        else:
            self.model.Gaussian_noise.constrain_bounded(1e-6,1e6, warning=False) #to avoid numerical problems


    def _init_deepgp(self):

        import socket
        self.useGPU = False
        if socket.gethostname()[0:4] == 'node':
            print 'Using GPU!'
            self.useGPU = True

        self.Ds = 1

        kern = [GPy.kern.Matern32(self.Ds, ARD=False), GPy.kern.Matern32(self.X.shape[1], ARD=False)]

        if self.model_type == 'deepgp_back_constraint':
            self.model = deepgp.DeepGP([self.Y.shape[1],self.Ds, self.X.shape[1]], self.Y, X=self.X, num_inducing=self.num_inducing, kernels=kern, MLP_dims=[[100,50],[]],repeatX=True)
        
        elif self.model_type == 'deepgp':
            self.model = deepgp.DeepGP([self.Y.shape[1],self.Ds, self.X.shape[1]], self.Y, X=self.X, num_inducing=self.num_inducing, kernels=kern, back_constraint=False,repeatX=True)

        if self.exact_feval == True:
            self.model.obslayer.Gaussian_noise.constrain_fixed(1e-6, warning=False) #to avoid numerical problems
        else:
            self.model.obslayer.Gaussian_noise.constrain_bounded(1e-6,1e6, warning=False) #to avoid numerical problems

    

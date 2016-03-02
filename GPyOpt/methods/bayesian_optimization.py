# Copyright (c) 2015, Javier Gonzalez
# Copyright (c) 2015, the GPy Authors (see GPy AUTHORS.txt)
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from __future__ import print_function

import GPy
import numpy as np
import warnings

from ..core.acquisition import AcquisitionEI
from ..core.acquisition import AcquisitionMPI
from ..core.acquisition import AcquisitionLCB
from ..core.acquisition import AcquisitionEL
from ..core.bo import BO
from ..util.general import samples_multidimensional_uniform
from ..util.general import reshape
from ..util.stats import initial_design

warnings.filterwarnings("ignore")


class BayesianOptimization(BO):
    def __init__(self, f, bounds=None, kernel=None, X=None, Y=None, numdata_initial_design = None,type_initial_design='random', model_optimize_interval=1, acquisition='EI',
        acquisition_par= 0.00, model_optimize_restarts=10, sparseGP=False, num_inducing=None, normalize=False,
        exact_feval=False, verbosity=0):
        '''
        Bayesian Optimization using EI, MPI and LCB (or UCB) acquisition functions.

        This is a thin wrapper around the methods.BO class, with a set of sensible defaults
        :param *f* the function to optimize. Should get a nxp numpy array as imput and return a nx1 numpy array.
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
        self.num_inducing = num_inducing
        self.sparseGP = sparseGP
        self.input_dim = len(bounds)
        self.normalize = normalize
        self.exact_feval = exact_feval
        self.model_optimize_interval = model_optimize_interval
        self.model_optimize_restarts = model_optimize_restarts
        self.verbosity = verbosity
        self.type_initial_design = type_initial_design

        if f==None:
            print('Function to optimize is required.')
        else:
            self.f = f

        # ------- Initialize model
        if bounds==None:
            raise Exception('Box constraints are needed. Please insert box constrains.')
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
            print('The selected acquisition function is not valid. Please try again with EI, MPI, or LCB')
        if (acquisition=='EI' or acquisition=='MPI' or acquisition =='LCB' or acquisition =='EL' ):
            super(BayesianOptimization ,self).__init__(acquisition_func=acq)


    def _init_model(self):
        '''
        Initializes the Gaussian Process over *f*.
        :param X: input observations.
        :param Y: output values.

        ..Note : X and Y can be None. In this case numdata_initial_design*input_dim data are uniformly generated to initialize the model.

        '''
        if self.sparseGP == True:
            if self.num_inducing ==None:
                raise Exception('Sparse model, please insert the number of inducing points')
            else:
                self.model = GPy.models.SparseGPRegression(self.X, self.Y, kernel=self.kernel, num_inducing=self.num_inducing)
        else:
            self.model = GPy.models.GPRegression(self.X,self.Y,kernel=self.kernel)

        if self.exact_feval == True:
            self.model.Gaussian_noise.constrain_fixed(1e-6, warning=False) #to avoid numerical problems
        else:
            self.model.Gaussian_noise.constrain_bounded(1e-6,1e6, warning=False) #to avoid numerical problems

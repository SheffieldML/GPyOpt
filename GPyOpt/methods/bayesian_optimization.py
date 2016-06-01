# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import GPy
import numpy as np
import time
from ..acquisitions import AcquisitionEI, AcquisitionMPI, AcquisitionLCB, AcquisitionEI_MCMC, AcquisitionMPI_MCMC, AcquisitionLCB_MCMC, AcquisitionLP  
from ..core.bo import BO
from ..core.task.space import Design_space, bounds_to_space
from ..core.task.objective import SingleObjective
from ..core.task.cost import CostModel
from ..util.general import samples_multidimensional_uniform, reshape, evaluate_function
from ..core.evaluators import Sequential, RandomBatch, Predictive, LocalPenalization
from ..util.stats import initial_design
from ..models.gpmodel import GPModel, GPModel_MCMC 
from ..models.rfmodel import RFModel
from ..models.warpedgpmodel import WarpedGPModel
from ..optimization.acquisition_optimizer import AcquisitionOptimizer
import GPyOpt

import warnings
warnings.filterwarnings("ignore")

class BayesianOptimization(BO):
    """
    Main class to initialize a Bayesian Optimization method.
    :param f: function to optimize. It should take 2-dimentional numpy arrays as input and return 2-dimentional outputs (one evaluation per row).
    :param domain: list of dictionies containin the description of the inputs variables (See GPyOpt.core.space.Design_space class for details).
    :param constrains: list of dicctionies containin the description of the problem constrains (See GPyOpt.core.space.Design_space class for details).
    :cost_withGradients: cost function of the objective. The input can be:
        - a fucntion that returns the cost and the derivatives and any set of points in the domain.
        - 'evaluation_time': a Gaussian process (mean) is used to handle the evaluation cost.
    :model_type: type of model to use as surrogate:
        - 'GP', standad Gaussian process.
        - 'GP_MCMC',  Gaussian process with priors in the hyperparameters.
        - 'sparseGP', sparse Gaussian process.
        - 'warperdGP', warped Gaussian process.
        - 'RF', random forrest (scikit-learn).    
    :param X: 2d numpy array containing the initial inputs (one per row) of the model.
    :param Y: 2d numpy array containing the initial outputs (one per row) of the model.
    :initial_design_numdata: number of initial points that are collected jointly before start running the optimization.
    :initial_design_type: type of initial design:
        - 'random', to collect points in random locations.
        - 'latin', to collect points in a latin hypercube (discrete variables are sampled randomly.)
    :acquisition_type: type of acquisition function to use.
        - 'EI', expected improvement.
        - 'EI_MCMC', integrated expected improvement (requieres GP_MCMC model).
        - 'MPI', maximum probability of improvement.
        - 'MPI_MCMC', maximum probability of improvement (requieres GP_MCMC model).
        - 'LCB', GP-Lower confidence bound.
        - 'LCB_MCMC', integrated GP-Lower confidence bound (requieres GP_MCMC model).
    :param normalize_Y: wheter to normalize the outputs before performing any optimization (default, True).
    :exact_feval: whether the outputs are exact (default, False).
    :acquisition_optimizer_type: type of acquisition function to use.
    :param model_update_interval: interval of collected observations after which the model is updated (default, 1). 
    :evaluator_type: determines the way the objective is evaluated (all methods are equivalent if the batch size is one)
        - 'sequential', sequential evaluations.
        - 'predictive', syncronous batch that uses a parallel model to phantasize new outputs.
        - 'random': syncronous batch that selects the firt element as in a sequential policy and the rest randomly.
        - 'local_penalization': batch method proposed in (Gonzalez et al. 2016). 
    :batch_size: size of the batch in which the objective is evaluted (default, 1).
    :num_cores: number of cores used to evaluate the objective (default, 1).
    :verbosity: prints the models and other options during the optimization.
    :**kwargs: extra parameters. Can be used to tune the current optimization setup or to use depreciated options in this package release.


    .. Note:: parameters used in previous versions of GPyOpt can be passed through the kwargs. This means that such parameters should be passed at the end. 
    """


    def __init__(self, f, domain = None, constrains = None, cost_withGradients = None, model_type = 'GP', X = None, Y = None, 
    	initial_design_numdata = None, initial_design_type='random', acquisition_type ='EI', normalize_Y = True, 
        exact_feval = False, acquisition_optimizer_type = 'lbfgs', model_update_interval=1, evaluator_type = 'sequential', 
        batch_size = 1, num_cores = 1, verbosity= True, **kwargs):

        self.initial_iter = True
        self.verbosity              = verbosity
        self.model_update_interval  = model_update_interval
        self.kwargs = kwargs

        # --- CHOOSE design space
        if domain == None and self.kwargs.has_key('bounds'): 
            self.domain = bounds_to_space(kwargs['bounds'])
        else: 
            self.domain = domain 
            
        self.constrains = constrains
        self.space = Design_space(self.domain, self.constrains)

        # --- CHOOSE objective function
        self.f = f
        if self.kwargs.has_key('objective_name'): self.objective_name = kwargs['objective_name']
        else: self.objective_name = 'no_name'  
        self.batch_size = batch_size
        self.num_cores = num_cores
        self.objective = SingleObjective(self.f, self.batch_size, self.num_cores,self.objective_name)

        # --- CHOOSE the cost model
        self.cost = CostModel(cost_withGradients)

        # --- CHOOSE initial design
        self.X = X
        self.Y = Y
        self.initial_design_type  = initial_design_type
        if initial_design_numdata==None: 
            self.initial_design_numdata = 5
        else: 
            self.initial_design_numdata = initial_design_numdata
        self._init_design_chooser()

        # --- CHOOSE the model type
        self.model_type = model_type  
        self.exact_feval = exact_feval  # note tha this 2 options are not used with the predefined model
        self.normalize_Y = normalize_Y      

        # If an istance of a GPyOpt model is passed (possibly user defined), it is used here:
        if self.kwargs.has_key('model'):
            if isinstance(kwargs['model'], GPyOpt.models.base.BOModel):
                self.model = kwargs['model']
                self.model_type = 'User defined model used.'
                print 'Using a model defined by the used.'
            else:
                self.model = self._model_chooser()
        else:
            self.model = self._model_chooser()

        # --- CHOOSE the acquisition optimizer_type
        self.acquisition_optimizer_type = acquisition_optimizer_type
        self.acquisition_optimizer = AcquisitionOptimizer(self.space, self.acquisition_optimizer_type, current_X = self.X)  ## morre arguments may come here

        # --- CHOOSE acquistion function
        self.acquisition_type = acquisition_type
        self.acquisition = self._acquisition_chooser()

        # --- CHOOSE evaluator method
        self.evaluator_type = evaluator_type
        self.evaluator = self._evaluator_chooser()

        # --- Create optimization space
        super(BayesianOptimization,self).__init__(	model                  = self.model, 
                									space                  = self.space, 
                									objective              = self.objective, 
                									acquisition            = self.acquisition, 
                                                    evaluator              = self.evaluator,
                									X_init                 = self.X, 
                                                    Y_init                 = self.Y,
                                                    cost                   = self.cost,
                									normalize_Y            = self.normalize_Y, 
                									model_update_interval  = self.model_update_interval)

        # --- Initilize everyting
        self.run_optimization(0,verbosity=self.verbosity)

    def _model_chooser(self):
        """
        Model chooser from the available options. Extra parameters can be passed via **kwargs.
        """
        
        if self.kwargs.has_key('kernel'): 
            self.kernel = self.kwargs['kernel']
        else: 
            self.kernel = None

        if self.kwargs.has_key('noise_var'): self.noise_var = self.kwargs['noise_var']
        else: self.noise_var = None
            
        # --------    
        # --- Initilize GP model with MLE on the parameters
        # --------
        if self.model_type == 'GP' or self.model_type == 'sparseGP':
            if self.kwargs.has_key('model_optimizer_type'): self.model_optimizer_type = self.kwargs['model_optimizer_type'] 
            else: self.model_optimizer_type = 'lbfgs' 

            if self.kwargs.has_key('optimize_restarts'): self.optimize_restarts = self.kwargs['optimize_restarts']
            else: self.optimize_restarts = 5

            if self.kwargs.has_key('max_iters'): self.max_iters = self.kwargs['max_iters']
            else: self.max_iters = 1000

            if self.kwargs.has_key('num_inducing'): self.num_inducing = self.kwargs['num_inducing']
            else: self.num_inducing = 10

            if self.model_type == 'GP': self.sparse = False
            if self.model_type == 'sparseGP': self.sparse = True

            return GPModel(self.kernel, self.noise_var, self.exact_feval, self.normalize_Y, self.model_optimizer_type, self.max_iters, self.optimize_restarts, self.sparse, self.num_inducing, self.verbosity)

        # --------
        # --- Initilize GP model with MCMC on the parameters
        # --------
        elif self.model_type == 'GP_MCMC':
            if self.kwargs.has_key('n_samples'): self.n_samples = self.kwargs['n_samples']
            else: self.n_samples = 10 

            if self.kwargs.has_key('n_burnin'): self.n_burnin = self.kwargs['n_burnin']
            else: self.n_burnin = 100
            
            if self.kwargs.has_key('subsample_interval'): self.subsample_interval = self.kwargs['subsample_interval']
            else: self.subsample_interval =10
            
            if self.kwargs.has_key('step_size'): self.step_size  = self.kwargs['step_size']
            else: self.step_size = 1e-1
            
            if self.kwargs.has_key('leapfrog_steps'): self.leapfrog_steps = self.kwargs['leapfrog_steps']
            else: self.leapfrog_steps = 20

            return  GPModel_MCMC(self.kernel, self.noise_var, self.exact_feval, self.normalize_Y, self.n_samples, self.n_burnin, self.subsample_interval, self.step_size, self.leapfrog_steps, self.verbosity)

        # --------
        # --- Initilize RF: values taken from default in scikit-learn
        # --------
        elif self.model_type =='RF':
            # TODO: add options via kwargs
            return RFModel(verbose=self.verbosity,  normalize_Y=self.normalize_Y)

        elif self.model_type =='warpedGP':
            return WarpedGPModel()



    def _acquisition_chooser(self):
        """
        Acquisition chooser from the available options. Extra parameters can be passed via **kwargs.
        """

        # --- Extract relevant parameters from the ***kwargs
        if self.kwargs.has_key('acquisition_jitter'):
            self.acquisition_jitter = self.kwargs['acquisition_jitter']
        else:
            self.acquisition_jitter = 0.01

        if self.kwargs.has_key('acquisition_weight'):
            self.acquisition_weight = self.kwargs['acquisition_weight']
        else:
            self.acquisition_weight = 2  ## TODO: implement the optimal rate (only for bandits)

        # --- Choose the acquisition
        if self.acquisition_type == None or self.acquisition_type =='EI':
            return AcquisitionEI(self.model, self.space, self.acquisition_optimizer, self.cost.cost_withGradients, self.acquisition_jitter)
        
        elif self.acquisition_type =='EI_MCMC':
            return AcquisitionEI_MCMC(self.model, self.space, self.acquisition_optimizer, self.cost.cost_withGradients, self.acquisition_jitter)        
        
        elif self.acquisition_type =='MPI':
            return AcquisitionMPI(self.model, self.space, self.acquisition_optimizer, self.cost.cost_withGradients, self.acquisition_jitter)
         
        elif self.acquisition_type =='MPI_MCMC':
            return AcquisitionMPI_MCMC(self.model, self.space, self.acquisition_optimizer, self.cost.cost_withGradients, self.acquisition_jitter)

        elif self.acquisition_type =='LCB':
            return AcquisitionLCB(self.model, self.space, self.acquisition_optimizer, self.cost.cost_withGradients, self.acquisition_weight)
        
        elif self.acquisition_type =='LCB_MCMC':
            return AcquisitionLCB_MCMC(self.model, self.space, self.acquisition_optimizer, self.cost.cost_withGradients, self.acquisition_weight)        
        
        else:
            raise Exception('Invalid acquisition selected.')


    def _init_design_chooser(self):
        """
        Initializes the choice of X and Y based on the selected initial design and number of points selected.
        """
        # Case 1:
        if self.X is None:
            self.X = initial_design(self.initial_design_type, self.space, self.initial_design_numdata)
            self.Y, _ = self.objective.evaluate(self.X)

        # Case 2
        elif self.X is not None and self.Y is None:
            self.Y, _ = self.objective.evaluate(self.X)


    def _evaluator_chooser(self):
        """
        Acquisition chooser from the available options. Guide the optimization through sequential or parallel evalutions of the objective.
        """

        if self.kwargs.has_key('acquisition_transformation'):
            self.acquisition_transformation = self.kwargs['acquisition_transformation']
        else:
            self.acquisition_transformation = 'none'

        if self.batch_size == 1 or self.evaluator_type == 'sequential':
            return Sequential(self.acquisition)

        elif self.batch_size >1 and (self.evaluator_type == 'random' or self.evaluator_type == None):
            return RandomBatch(self.acquisition, self.batch_size)

        elif self.evaluator_type == 'predictive':
            return Predictive(self.acquisition, self.batch_size,self.normalize_Y)

        elif self.evaluator_type == 'local_penalization':
            if not isinstance(self.acquisition, AcquisitionLP):
                self.acquisition = AcquisitionLP(self.model, self.space, self.acquisition_optimizer, self.acquisition, self.acquisition_transformation)
            return LocalPenalization(self.acquisition, self.batch_size, self.normalize_Y)






    

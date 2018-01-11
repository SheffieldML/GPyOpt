# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from ..acquisitions import AcquisitionEI, AcquisitionMPI, AcquisitionLCB, AcquisitionEI_MCMC, AcquisitionMPI_MCMC, AcquisitionLCB_MCMC, AcquisitionLP
from ..core.bo import BO
from ..core.errors import InvalidConfigError
from ..core.task.space import Design_space, bounds_to_space
from ..core.task.objective import SingleObjective
from ..core.task.cost import CostModel
from ..experiment_design import initial_design
from ..util.arguments_manager import ArgumentsManager
from ..core.evaluators import Sequential, RandomBatch, LocalPenalization, ThompsonBatch
from ..models.gpmodel import GPModel, GPModel_MCMC
from ..models.rfmodel import RFModel
from ..models.warpedgpmodel import WarpedGPModel
from ..models.input_warped_gpmodel import InputWarpedGPModel
from ..optimization.acquisition_optimizer import AcquisitionOptimizer
import GPyOpt

import warnings
warnings.filterwarnings("ignore")

class BayesianOptimization(BO):
    """
    Main class to initialize a Bayesian Optimization method.
    :param f: function to optimize. It should take 2-dimensional numpy arrays as input and return 2-dimensional outputs (one evaluation per row).
    :param domain: list of dictionaries containing the description of the inputs variables (See GPyOpt.core.space.Design_space class for details).
    :param constraints: list of dictionaries containing the description of the problem constraints (See GPyOpt.core.space.Design_space class for details).
    :cost_withGradients: cost function of the objective. The input can be:
        - a function that returns the cost and the derivatives and any set of points in the domain.
        - 'evaluation_time': a Gaussian process (mean) is used to handle the evaluation cost.
    :model_type: type of model to use as surrogate:
        - 'GP', standard Gaussian process.
        - 'GP_MCMC',  Gaussian process with prior in the hyper-parameters.
        - 'sparseGP', sparse Gaussian process.
        - 'warperdGP', warped Gaussian process.
        - 'InputWarpedGP', input warped Gaussian process
        - 'RF', random forest (scikit-learn).
    :param X: 2d numpy array containing the initial inputs (one per row) of the model.
    :param Y: 2d numpy array containing the initial outputs (one per row) of the model.
    :initial_design_numdata: number of initial points that are collected jointly before start running the optimization.
    :initial_design_type: type of initial design:
        - 'random', to collect points in random locations.
        - 'latin', to collect points in a Latin hypercube (discrete variables are sampled randomly.)
    :acquisition_type: type of acquisition function to use.
        - 'EI', expected improvement.
        - 'EI_MCMC', integrated expected improvement (requires GP_MCMC model).
        - 'MPI', maximum probability of improvement.
        - 'MPI_MCMC', maximum probability of improvement (requires GP_MCMC model).
        - 'LCB', GP-Lower confidence bound.
        - 'LCB_MCMC', integrated GP-Lower confidence bound (requires GP_MCMC model).
    :param normalize_Y: whether to normalize the outputs before performing any optimization (default, True).
    :exact_feval: whether the outputs are exact (default, False).
    :acquisition_optimizer_type: type of acquisition function to use.
        - 'lbfgs': L-BFGS.
        - 'DIRECT': Dividing Rectangles.
        - 'CMA': covariance matrix adaptation.
    :param model_update_interval: interval of collected observations after which the model is updated (default, 1).
    :param evaluator_type: determines the way the objective is evaluated (all methods are equivalent if the batch size is one)
        - 'sequential', sequential evaluations.
        - 'random': synchronous batch that selects the first element as in a sequential policy and the rest randomly.
        - 'local_penalization': batch method proposed in (Gonzalez et al. 2016).
        - 'thompson_sampling': batch method using Thompson sampling.
    :param batch_size: size of the batch in which the objective is evaluated (default, 1).
    :param num_cores: number of cores used to evaluate the objective (default, 1).
    :param verbosity: prints the models and other options during the optimization (default, False).
    :param maximize: when True -f maximization of f is done by minimizing -f (default, False).
    :param **kwargs: extra parameters. Can be used to tune the current optimization setup or to use deprecated options in this package release.


    .. Note::   The parameters bounds, kernel, numdata_initial_design, type_initial_design, model_optimize_interval, acquisition, acquisition_par
                model_optimize_restarts, sparseGP, num_inducing and normalize can still be used but will be deprecated in the next version.
    """

    def __init__(self, f, domain = None, constraints = None, cost_withGradients = None, model_type = 'GP', X = None, Y = None,
    	initial_design_numdata = 5, initial_design_type='random', acquisition_type ='EI', normalize_Y = True,
        exact_feval = False, acquisition_optimizer_type = 'lbfgs', model_update_interval=1, evaluator_type = 'sequential',
        batch_size = 1, num_cores = 1, verbosity=False, verbosity_model = False, maximize=False, de_duplication=False, **kwargs):

        self.modular_optimization = False
        self.initial_iter = True
        self.verbosity = verbosity
        self.verbosity_model = verbosity_model
        self.model_update_interval = model_update_interval
        self.de_duplication = de_duplication
        self.kwargs = kwargs

        # --- Handle the arguments passed via kargs
        self.problem_config = ArgumentsManager(kwargs)

        # --- CHOOSE design space
        self.constraints = constraints
        self.domain = domain
        self.space = Design_space(self.domain, self.constraints)

        # --- CHOOSE objective function
        self.maximize = maximize
        if 'objective_name' in kwargs: self.objective_name = kwargs['objective_name']
        else: self.objective_name = 'no_name'
        self.batch_size = batch_size
        self.num_cores = num_cores
        if f is not None:
            self.f = self._sign(f)
            self.objective = SingleObjective(self.f, self.batch_size,self.objective_name)
        else:
            self.f = None
            self.objective = None

        # --- CHOOSE the cost model
        self.cost = CostModel(cost_withGradients)

        # --- CHOOSE initial design
        self.X = X
        self.Y = Y
        self.initial_design_type  = initial_design_type
        self.initial_design_numdata = initial_design_numdata
        self._init_design_chooser()

        # --- CHOOSE the model type. If an instance of a GPyOpt model is passed (possibly user defined), it is used.
        self.model_type = model_type
        self.exact_feval = exact_feval  # note that this 2 options are not used with the predefined model
        self.normalize_Y = normalize_Y

        if 'model' in self.kwargs:
            if isinstance(kwargs['model'], GPyOpt.models.base.BOModel):
                self.model = kwargs['model']
                self.model_type = 'User defined model used.'
                print('Using a model defined by the used.')
            else:
                self.model = self._model_chooser()
        else:
            self.model = self._model_chooser()

        # --- CHOOSE the acquisition optimizer_type

        # This states how the discrete variables are handled (exact search or rounding)
        self.acquisition_optimizer_type = acquisition_optimizer_type
        self.acquisition_optimizer = AcquisitionOptimizer(self.space, self.acquisition_optimizer_type, model=self.model)  ## more arguments may come here

        # --- CHOOSE acquisition function. If an instance of an acquisition is passed (possibly user defined), it is used.
        self.acquisition_type = acquisition_type

        if 'acquisition' in self.kwargs:
            if isinstance(kwargs['acquisition'], GPyOpt.acquisitions.AcquisitionBase):
                self.acquisition = kwargs['acquisition']
                self.acquisition_type = 'User defined acquisition used.'
                print('Using an acquisition defined by the used.')
            else:
                self.acquisition = self._acquisition_chooser()
        else:
            self.acquisition = self.acquisition = self._acquisition_chooser()


        # --- CHOOSE evaluator method
        self.evaluator_type = evaluator_type
        self.evaluator = self._evaluator_chooser()

        # --- Create optimization space
        super(BayesianOptimization,self).__init__(  model                  = self.model,
                                                    space                  = self.space,
                                                    objective              = self.objective,
                                                    acquisition            = self.acquisition,
                                                    evaluator              = self.evaluator,
                                                    X_init                 = self.X,
                                                    Y_init                 = self.Y,
                                                    cost                   = self.cost,
                                                    normalize_Y            = self.normalize_Y,
                                                    model_update_interval  = self.model_update_interval,
                                                    de_duplication         = self.de_duplication)

    def _model_chooser(self):
        return self.problem_config.model_creator(self.model_type, self.exact_feval,self.space)

    def _acquisition_chooser(self):
        return self.problem_config.acquisition_creator(self.acquisition_type, self.model, self.space, self.acquisition_optimizer, self.cost.cost_withGradients)

    def _evaluator_chooser(self):
        return self.problem_config.evaluator_creator(self.evaluator_type, self.acquisition, self.batch_size, self.model_type, self.model, self.space, self.acquisition_optimizer)

    def _init_design_chooser(self):
        """
        Initializes the choice of X and Y based on the selected initial design and number of points selected.
        """

        # If objective function was not provided, we require some initial sample data
        if self.f is None and (self.X is None or self.Y is None):
            raise InvalidConfigError("Initial data for both X and Y is required when objective function is not provided")

        # Case 1:
        if self.X is None:
            self.X = initial_design(self.initial_design_type, self.space, self.initial_design_numdata)
            self.Y, _ = self.objective.evaluate(self.X)
        # Case 2
        elif self.X is not None and self.Y is None:
            self.Y, _ = self.objective.evaluate(self.X)

    def _sign(self,f):
         if self.maximize:
             f_copy = f
             def f(x):return -f_copy(x)
         return f

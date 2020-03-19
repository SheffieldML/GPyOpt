from ..models.gpmodel import GPModel, GPModel_MCMC
from ..models.rfmodel import RFModel
from ..models.warpedgpmodel import WarpedGPModel
from ..models.input_warped_gpmodel import InputWarpedGPModel
from ..core.evaluators import Sequential, RandomBatch, LocalPenalization, ThompsonBatch
from ..acquisitions import AcquisitionEI, AcquisitionMPI, AcquisitionLCB, AcquisitionEI_MCMC, AcquisitionMPI_MCMC, AcquisitionLCB_MCMC, AcquisitionLP
from ..core.errors import InvalidConfigError

class ArgumentsManager(object):
    """
    Class to handle extra configurations in the definition of the BayesianOptimization class
    """
    def __init__(self,kwargs):
        self.kwargs = kwargs


    def evaluator_creator(self, evaluator_type, acquisition, batch_size, model_type, model, space, acquisition_optimizer):
        """
        Acquisition chooser from the available options. Guide the optimization through sequential or parallel evalutions of the objective.
        """
        acquisition_transformation = self.kwargs.get('acquisition_transformation','none')

        if batch_size == 1 or evaluator_type == 'sequential':
            return Sequential(acquisition)

        elif batch_size >1 and (evaluator_type == 'random' or evaluator_type is  None):
            return RandomBatch(acquisition, batch_size)

        elif batch_size >1 and evaluator_type == 'thompson_sampling':
            return ThompsonBatch(acquisition, batch_size)

        elif evaluator_type == 'local_penalization':
            if model_type not in ['GP', 'sparseGP', 'GP_MCMC', 'warpedGP']:
                raise InvalidConfigError('local_penalization evaluator can only be used with GP models')

            if not isinstance(acquisition, AcquisitionLP):
                acquisition_lp = AcquisitionLP(model, space, acquisition_optimizer, acquisition, acquisition_transformation)
            return LocalPenalization(acquisition_lp, batch_size)



    def acquisition_creator(self, acquisition_type, model, space, acquisition_optimizer, cost_withGradients):

        """
        Acquisition chooser from the available options. Extra parameters can be passed via **kwargs.
        """
        acquisition_type = acquisition_type
        model = model
        space = space
        acquisition_optimizer = acquisition_optimizer
        cost_withGradients = cost_withGradients
        acquisition_jitter = self.kwargs.get('acquisition_jitter',0.01)
        acquisition_weight = self.kwargs.get('acquisition_weight',2)

        # --- Choose the acquisition
        if acquisition_type is  None or acquisition_type =='EI':
            return AcquisitionEI(model, space, acquisition_optimizer, cost_withGradients, acquisition_jitter)

        elif acquisition_type =='EI_MCMC':
            return AcquisitionEI_MCMC(model, space, acquisition_optimizer, cost_withGradients, acquisition_jitter)

        elif acquisition_type =='MPI':
            return AcquisitionMPI(model, space, acquisition_optimizer, cost_withGradients, acquisition_jitter)

        elif acquisition_type =='MPI_MCMC':
            return AcquisitionMPI_MCMC(model, space, acquisition_optimizer, cost_withGradients, acquisition_jitter)

        elif acquisition_type =='LCB':
            return AcquisitionLCB(model, space, acquisition_optimizer, None, acquisition_weight)

        elif acquisition_type =='LCB_MCMC':
            return AcquisitionLCB_MCMC(model, space, acquisition_optimizer, None, acquisition_weight)

        else:
            raise Exception('Invalid acquisition selected.')


    def model_creator(self, model_type, exact_feval, space):
        """
        Model chooser from the available options. Extra parameters can be passed via **kwargs.
        """
        model_type = model_type
        exact_feval = exact_feval
        space = space

        kernel = self.kwargs.get('kernel',None)
        ARD = self.kwargs.get('ARD',False)
        verbosity_model = self.kwargs.get('verbosity_model',False)
        noise_var = self.kwargs.get('noise_var',None)
        model_optimizer_type = self.kwargs.get('model_optimizer_type','lbfgs')
        max_iters = self.kwargs.get('max_iters',1000)
        optimize_restarts = self.kwargs.get('optimize_restarts',5)

        # --------
        # --- Initialize GP model with MLE on the parameters
        # --------
        if model_type == 'GP' or model_type == 'sparseGP':
            if model_type == 'GP':
                sparse = False
            if model_type == 'sparseGP':
                sparse = True
            optimize_restarts = self.kwargs.get('optimize_restarts',5)
            num_inducing = self.kwargs.get('num_inducing',10)
            return GPModel(kernel, noise_var, exact_feval, model_optimizer_type, max_iters, optimize_restarts, sparse, num_inducing, verbosity_model, ARD)

        # --------
        # --- Initialize GP model with MCMC on the parameters
        # --------
        elif model_type == 'GP_MCMC':
            n_samples = self.kwargs.get('n_samples',10)
            n_burnin = self.kwargs.get('n_burnin',100)
            subsample_interval = self.kwargs.get('subsample_interval',10)
            step_size = self.kwargs.get('step_size',1e-1)
            leapfrog_steps = self.kwargs.get('leapfrog_steps',20)
            return GPModel_MCMC(kernel, noise_var, exact_feval, n_samples, n_burnin, subsample_interval, step_size, leapfrog_steps, verbosity_model)

        # --------
        # --- Initialize RF: values taken from default in scikit-learn
        # --------
        elif model_type =='RF':
            return RFModel(verbose=verbosity_model)

        # --------
        # --- Initialize WapedGP in the outputs
        # --------
        elif model_type =='warpedGP':
            return WarpedGPModel()

        # --------
        # --- Initialize WapedGP in the inputs
        # --------
        elif model_type == 'input_warped_GP':
            if 'input_warping_function_type' in self.kwargs:
                if self.kwargs['input_warping_function_type'] != "kumar_warping":
                    print("Only support kumar_warping for input!")

            # Only support Kumar warping now, setting it to None will use default Kumar warping
            input_warping_function = None
            optimize_restarts = self.kwargs.get('optimize_restarts',5)
            return InputWarpedGPModel(space, input_warping_function, kernel, noise_var,
                                      exact_feval, model_optimizer_type, max_iters,
                                      optimize_restarts, verbosity_model, ARD)

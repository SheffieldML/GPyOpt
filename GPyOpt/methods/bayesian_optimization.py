# Copyright (c) 2015, Javier Gonzalez
# Copyright (c) 2015, the GPy Authors (see GPy AUTHORS.txt)
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import GPy
import deepgp
import numpy as np
import time
from ..core.acquisition import AcquisitionEI, AcquisitionMPI, AcquisitionLCB, AcquisitionEL
from ..core.bo import BO
from ..core.task.space import Design_space, bounds_to_space
from ..core.task.objective import SingleObjective
from ..util.general import samples_multidimensional_uniform, reshape, evaluate_function
from ..util.stats import initial_design


import warnings
warnings.filterwarnings("ignore")


class BayesianOptimization(BO):

    def __init__(self, f, domain = None, constrains = None, cost_withGradients = None, model_type = 'gp', X = None, Y = None, 
    	initial_design_numdata = None, initial_design_type='random', acquisition_type ='EI', normalize_Y = True, 
        exact_feval = False, verbosity=0, **kargs):

        self.verbosity              = verbosity
        self.model_update_interval  = model_update_interval

        # --- CHOOSE design space
        if self.domain == None and self.kargs.has_key('bounds'):
            self.domain             = bounds_to_space(kargs['bounds'])
        else:
            self.domain = domain 

        self.constrains = constrains
        self.space = Design_space(self.space, self.constrains)

        # --- CHOOSE objective function
        self.f                   = f
        self.cost_withGradients = cost_withGradients
        self.objective          = SingleObjective(self.f, self.space, self.cost_withGradients)

        # --- CHOOSE initial design
        self.X = X
        self.Y = Y
        self.initial_design_type    = initial_design_type
        self.initial_design_numdata = initial_design_numdata
        self._init_design_chooser()

        # --- CHOOSE the model type
        self.model_type         = model_type  
        self.exact_feval        = exact_feval 
        self.normalize_Y        = normalize_Y      
        self.model              = self._model_chooser()

        # --- CHOOSE the acquisition optimizer
        self.aquisition_optimizer = GPyOpt.optimization.ContAcqOptimizer(space, 1000, search=True)

        # --- CHOOSE acquistion function
        self.acquisition_type       = acquisition_type
        self.acquisition            = self.acquisition_chooser()

        # -- Create optimization space
        super(BayesianOptimization ,self).__init__(	model                  = self.model, 
                									space                  = self.space, 
                									objective              = self.objective, 
                									acquisition_func       = self.acquisition, 
                									X_init                 = self.X, 
                                                    Y_init                 = self.Y,
                									normalize_Y            = self.normalize_Y, 
                									model_update_interval  = self.model_update_interval)


	def _model_chooser(self):

        # --- extra arguments defined in **kargs
        if self.kargs.has_key('kernel'): self.kernel = kargs['kernel']
        else: self.kernel = None

        if self.kargs.has_key('noise_var'): self.noise_var = kargs['noise_var']
        else: self.noise_var = None
            
        # --- Initilize GP model with MLE on the parameters
        if self.model_type == 'gp':
            if self.kargs.has_key('optimizer'): self.optimizer = kargs['optimizer'] 
            else: self.optimizer = 'bfgs' 

            if self.kargs.has_key('optimize_restarts'): self.optimize_restarts = kargs['optimize_restarts']
            else: self.optimize_restarts = 5

            if self.kargs.has_key('max_iters'): self.max_iters = kargs['max_iters']
            else: self.max_iters = 1000 

            self.model =  GPModel(self.kernel, self.noise_var, self.exact_feval, self.normalize_Y, self.optimizer, self.max_iters, self.optimize_restarts, self.verbose)

        # --- Initilize GP model with MCMC on the parameters
        elif self.model_type == 'gp_mcmc':
            if self.kargs.has_key('n_samples'): kargs['n_samples']
            else: self.n_samples = 10 

            if self.kargs.has_key('n_burnin'): kargs['n_burnin']
            else: self.n_burnin = 100
            
            if self.kargs.has_key('subsample_interval'): kargs['subsample_interval']
            else: self.subsample_interval =10
            
            if self.kargs.has_key('step_size'): kargs['step_size']
            else: self.step_size = 1e-1
            
            if self.kargs.has_key('leapfrog_steps'): kargs['leapfrog_steps']
            else: self.leapfrog_steps = 20

            self.model =  GPModel_MCMC(self.kernel, self.noise_var, self.exact_feval, self.normalize_Y, self.n_samples, self.n_burnin, self.subsample_interval, self.step_size, self.leapfrog_steps, self.verbose)



    def _acquisition_chooser(self):

        # --- Extract relevant parameters from the ***kargs
        if self.kargs.has_key('acquisition_jitter'):
            self.acquisition_jitter = kargs['acquisition_jitter']
        else:
            self.acquisition_jitter = 0.01

        if self.kargs.has_key('acquisition_weight'):
            self.acquisition_weight = kargs['acquisition_weight']
        else:
            self.acquisition_weight = 2  ## TODO: implement the optimal rate (only for bandits)

        # --- Choose the acquisition
        elif self.acquisition_type == None or acquisition_type =='EI':
            return AcquisitionEI(self.model, self.space, self.optimizer, self.cost, self.acquisition_jitter)
        
        elif self.acquisition_type =='EI_MCMC':
            return AcquisitionEI_MCMC(self.model, self.space, self.optimizer, self.cost, self.acquisition_jitter)        
        
        elif self.acquisition_type =='MPI':
            return AcquisitionMPI(self.model, self.space, self.optimizer, self.cost, self.acquisition_jitter)
         
        elif self.acquisition_type =='MPI_MCMC':
            return AcquisitionMPI_MCMC(self.model, self.space, self.optimizer, self.cost, self.acquisition_jitter)

        elif self.acquisition_type =='LCB':
            return AcquisitionLCB(self.model, self.space, self.optimizer, cself.cost, self.acquisition_weight)
        
        elif self.acquisition_type =='LCB_MCMC':
            return AcquisitionLCB_MCMC(self.model, self.space, self.optimizer, self.cost, self.acquisition_weight)        
        
        else:
            raise Exception('Invalid acquisition selected.')


    def _init_design_chooser(self):
        if self.initial_design_numdata==None: 
            self.initial_design_numdata = 5
        else: self.initial_design_numdata = initial_design_numdata

        # Case 1:
        if self.X is None:
            self.X = initial_design(self.initial_design_type, self.space, self.initial_design_numdata)
            self.Y, _ = self.objective.evaluate(self.X)

        # Case 2
        elif self.X is not None and self.Y is None:
            self.Y, _ = self.objective.evaluate(self.X)




    

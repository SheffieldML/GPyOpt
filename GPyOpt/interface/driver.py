# Copyright (c) 2014, GPyOpt authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
import time
from ..methods import BayesianOptimization

class BODriver(object):
    """
    The class for driving the Bayesian optimization according to the configuration.
    """
    
    def __init__(self, config=None, obj_func=None, outputEng=None):
        
        if config is None:
            from .config_parser import default_config
            import copy
            self.config = copy.deepcopy(default_config)
        else:
            self.config = config
        self.obj_func = obj_func
        self.outputEng = outputEng
        
    def _get_obj(self,space):

        """
        Imports the acquisition function.
        """
        obj_func = self.obj_func
        
        from ..core.task import SingleObjective
        return SingleObjective(obj_func, self.config['resources']['cores'], space=space, unfold_args=True)
        
    def _get_space(self):
        """
        Imports the domain.
        """
        assert 'space' in self.config, 'The search space is NOT configured!'
        
        space_config = self.config['space']
        constraint_config = self.config['constraints']
        from ..core.task.space import Design_space
        return Design_space.fromConfig(space_config, constraint_config)
    
    def _get_model(self):
        """
        Imports the model.
        """

        from copy import deepcopy
        model_args = deepcopy(self.config['model'])
        del model_args['type']
        from ..models import select_model
        
        return select_model(self.config['model']['type']).fromConfig(model_args)
        
    
    def _get_acquisition(self, model, space):
        """
        Imports the acquisition
        """

        from copy import deepcopy        
        acqOpt_config = deepcopy(self.config['acquisition']['optimizer'])
        acqOpt_name = acqOpt_config['name']
        del acqOpt_config['name']
        
        from ..optimization import AcquisitionOptimizer
        acqOpt = AcquisitionOptimizer(space, acqOpt_name, **acqOpt_config)
        from ..acquisitions import select_acquisition
        return select_acquisition(self.config['acquisition']['type']).fromConfig(model, space, acqOpt, None, self.config['acquisition'])
    
    def _get_acq_evaluator(self, acq):
        """
        Imports the evaluator
        """

        from ..core.evaluators import select_evaluator
        from copy import deepcopy
        eval_args = deepcopy(self.config['acquisition']['evaluator'])
        del eval_args['type']
        return select_evaluator(self.config['acquisition']['evaluator']['type'])(acq, **eval_args)
    
    def _check_stop(self, iters, elapsed_time, converged):
        """
        Defines the stopping criterion.
        """

        r_c = self.config['resources']
        
        stop = False
        if converged==0:
            stop=True
        if r_c['maximum-iterations'] !='NA' and iters>= r_c['maximum-iterations']:
            stop = True
        if r_c['max-run-time'] != 'NA' and elapsed_time/60.>= r_c['max-run-time']:
            stop = True
        return stop
            
    def run(self):
        """
        Runs the optimization using the previously loaded elements.
        """

        space = self._get_space()
        obj_func = self._get_obj(space)
        model = self._get_model()
        acq = self._get_acquisition(model, space)
        acq_eval = self._get_acq_evaluator(acq)
                
        from ..experiment_design import initial_design
        X_init = initial_design(self.config['initialization']['type'], space, self.config['initialization']['num-eval'])

        from ..methods import ModularBayesianOptimization
        bo = ModularBayesianOptimization(model, space, obj_func, acq, acq_eval, X_init)
                
        bo.run_optimization(max_iter = self.config['resources']['maximum-iterations'], max_time = self.config['resources']['max-run-time'] if self.config['resources']['max-run-time']!="NA" else np.inf,
                              eps = self.config['resources']['tolerance'], verbosity=True)        
        return bo

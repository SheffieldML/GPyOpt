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
        
    def _get_obj(self):
        obj_func = self.obj_func.objective_function()
        
        from ..core.task import SingleObjective
        return SingleObjective(obj_func, self.config['resources']['cores'])
        
    def _get_space(self):
        assert 'space' in self.config, 'The search space is NOT configured!'
        
        space_config = self.config['space']
        constraint_config = None if len(self.config['constraints'])==0 else self.config['constraints']  
        from ..core.task.space import Design_space
        return Design_space(space_config, constraint_config)
    
    def _get_model(self):

        from copy import deepcopy
        model_args = deepcopy(self.config['model'])
        del model_args['type']
        from ..models import select_model
        
        return select_model(self.config['model']['type']).fromConfig(model_args)
        
    
    def _get_acquisition(self, model, space):

        from copy import deepcopy        
        acqOpt_config = deepcopy(self.config['acquisition']['optimizer'])
        acqOpt_name = acqOpt_config['name']
        del acqOpt_config['name']
        
        from ..optimization import AcquisitionOptimizer
        acqOpt = AcquisitionOptimizer(space, acqOpt_name, **acqOpt_config)
        from ..acquisitions import select_acquisition
        return select_acquisition(self.config['acqusition']['type']).fromConfig(model, space, acqOpt, None, self.config['acqusition'])
    
    def _get_acq_evaluator(self, acq):
        from ..core.evaluators import select_evaluator
        from copy import deepcopy
        eval_args = deepcopy(self.config['acquisition']['evaluator'])
        del eval_args['type']
        return select_evaluator(self.config['acquisition']['evaluator']['type'], **eval_args)
    
    def _check_stop(self, iters, elapsed_time, converged):
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
        space = self._get_space()
        obj_func = self._get_obj()
        model = self._get_model()
        acq = self._get_acquisition(model, space)
        acq_eval = self._get_acq_evaluator(acq)
        
        from ..methods import ModularBayesianOptimization
        bo = ModularBayesianOptimization(model, space, obj_func, acq, acq_eval, None)
        
        bo.run_optimization(max_iter = self.config['resources']['maximum-iterations'], max_time = self.config['resources']['max-run-time'] if self.config['resources']['max-run-time']!="NA" else np.inf,
                              eps = self.config['resources']['tolerance'], verbosity=True)        
        return bo

# Copyright (c) 2014, GPyOpt authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

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
        
    def _get_bounds(self):
        assert 'variables' in self.config, 'No variable configurations!'
        
        bounds = []
        var = self.config['variables']
        # for k in var.keys():
        for k in list(var.keys()):
            assert var[k]['type'].lower().startswith('float'), 'Only real value variables are supported!'
            bounds.extend([(float(var[k]['min']), float(var[k]['max']))]*int(var[k]['size']))
        return bounds
    
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
        m_c, a_c, r_c, p_c = self.config['model'], self.config['acquisition'], self.config['resources'], self.config['parallelization']
        o_c = self.config['output']
        bounds = self._get_bounds()
        obj_func = self.obj_func.objective_function()
        
        xs_init = None
        ys_init = None
        iters = 0
        offset = 0
    
        bo = BayesianOptimization(obj_func, bounds=bounds, X= xs_init, Y=ys_init, 
                                                 numdata_initial_design = m_c['initial-points'],type_initial_design= m_c['design-initial-points'],
                                                 model_optimize_interval=m_c['optimization-interval'],model_optimize_restarts=m_c['optimization-restarts'],
                                                 sparseGP=True if m_c['type'].lower()=='sparsegp' else False, num_inducing=m_c['inducing-points'],
                                                 acquisition=a_c['type'], acquisition_par = a_c['parameter'],normalize=m_c['normalized-evaluations'],
                                                 exact_feval=True if self.config['likelihood'].lower()=="noiseless" else False, verbosity=o_c['verbosity'])
        X, Y = bo.get_evaluations()
        offset = X.shape[0]
        if self.outputEng is not None: self.outputEng.append_iter(iters, 0., X, Y, bo)

        start_time = time.time()
        while True:
            rt = bo.run_optimization(max_iter = r_c['iterations_per_call'], n_inbatch= p_c['batch-size'], batch_method = p_c['type'], 
                                      acqu_optimize_method=a_c['optimization-method'], acqu_optimize_restarts= a_c['optimization-restarts'], 
                                      true_gradients = a_c["true-gradients"], n_procs=r_c['cores'], eps=r_c['tolerance'], verbose=o_c['verbosity'])
            
            iters += r_c['iterations_per_call']
            elapsed_time = time.time() - start_time

            X, Y = bo.get_evaluations()
            if self.outputEng is not None:  self.outputEng.append_iter(iters, elapsed_time, X[offset:], Y[offset:], bo)
            offset = X.shape[0]
            
            if self._check_stop(iters, elapsed_time, rt): break
        self.outputEng.append_iter(iters, elapsed_time, X[offset:], Y[offset:], bo, final=True)
        return bo

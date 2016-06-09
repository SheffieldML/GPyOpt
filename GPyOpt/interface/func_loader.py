# Copyright (c) 2014, GPyOpt authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import os
import numpy as np
from builtins import range  # compatible with 2 and 3

class ObjectiveFunc(object):
    
    def __init__(self, config):
        self.orgfunc, self.support_multi_eval = self._load_func(config)
        self.param_names, self.param_sizes, self.param_offsets, self.param_types, self.total_size = self._load_param_config(config)
        self.func = self._create_obj_func()
        
    def _load_func(self, config):
    
        assert 'prjpath' in config
        assert 'main-file' in config, "The problem file ('main-file') is missing!"
        
        os.chdir(config['prjpath'])
        if config['language'].lower()=='python':
            assert config['main-file'].endswith('.py'), 'The python problem file has to end with .py!'
#            m = __import__(config['main-file'][:-3])
            import imp
            m = imp.load_source(config['main-file'][:-3], os.path.join(config['prjpath'],config['main-file']))
            func = m.__dict__[config['main-file'][:-3]]
        return func, config['support-multi-eval']
    
    def _load_param_config(self, config):
        
        assert 'variables' in config
        var = config['variables']
        
        param_names = []
        param_sizes = []
        param_offsets = []
        param_types = []
        total_size = 0
        for k, v in list(var.items()):
            param_names.append(k)
            param_sizes.append(v['size'])
            assert v['type'].lower().startswith('float'), 'Only real parameters are supported at the moment!'
            param_types.append(v['type'])
            param_offsets.append(total_size)
            total_size += v['size']
        return param_names, param_sizes, param_offsets, param_types, total_size
    
    def _create_obj_func(self):
        def obj_func(x):
            params = []
            if len(x.shape)==1: x = x[None,:]
            for i in range(len(self.param_sizes)):
                params.append(x[:,self.param_offsets[i]:self.param_offsets[i]+self.param_sizes[i]])
            
            if self.support_multi_eval:
                return self.orgfunc(*params)
            else:
                rts = np.empty((x.shape[0],1))
                for i in range(x.shape[0]):
                    rts[i] = self.orgfunc(*[p[i] for p in params])
                return rts
        return obj_func
            
    def objective_function(self):
        return self.func

import numpy as np
import itertools

# class Objective(object):
#     """
#     This is a class to handle the objective function, its associated cost and the way it should be evaluated.

#     - func: is a list of functions to be optimized. It has only one element in the most general case.
#     - cost: is a lisy of costs of evaluating the functions to optimize. In principle it will be a deterministic function but 
#     we can extend this and handle it with the log of a GP.
#     - evaluation_type: determine how the function is going to be evaluated ['sequentiely','syncr_batch','async_batch']

#     """
#     def __init__(self, func, cost = None, evaluation_type = None):
#         if len(func)==1:
#             self.type = 'standard' 
#             self.f = {'f1': func}
            
#         else: 
#             self.type = 'multi-task'
#             self.f = {}
#             for i in len(func):
#                 self.f['f'+str(i+1)] = func[i]


class Design_space(object):
    """
    The format of a input domain:
    a list of dictionaries, one for each dimension, contains a list of attributes, e.g.:
    [   {'name': 'var_1', 'type': 'continuous', 'domain':(-1,1), 'dimensionality':1},
        {'name': 'var_2', 'type': 'discrete', 'domain': (0,1,2,3)},
        {'name': 'var_3', 'type': 'bandit', 'domain': [(-1,1),(1,0),(0,1)], 'dimensionality':2} ]
    """
    
    supported_types = ['continuous', 'discrete', 'bandit']
    
    def __init__(self, space):
        self._complete_attributes(space)
        
    def _complete_attributes(self, space):
        from copy import deepcopy
        self.space = []
        self.dimensionality = 0
        self.has_types = d = {type: False for type in self.supported_types}
        for i in range(len(space)):
            d_out = deepcopy(space[i])
            if 'name' not in d_out:
                d_out['name'] = 'var_'+str(i+1)
            if 'type' not in d_out:
                d_out['type'] = 'continuous'
            if 'domain' not in d_out:
                raise Exception('Domain attribute cannot be missing!')
            if 'dimensionality' not in d_out:
                d_out['dimensionality'] = 1
            self.dimensionality += d_out['dimensionality']
            self.space.append(d_out)
            self.has_types[d_out['type']] = True

    def get_continuous_bounds(self):
        bounds = []
        for d in self.space:
            if d['type']=='continuous':
                bounds.extend([d['domain']]*d['dimensionality'])
        return bounds
    
    def get_discrete_grid(self):
        sets_grid = []
        for d in self.space:
            if d['type']=='discrete':
                sets_grid.extend([d['domain']]*d['dimensionality'])
        return np.array(list(itertools.product(*sets_grid)))

    def get_bandit(self):
        bandit = self.space['domain'][sef.space['type'==bandit]]
        return np.array(bandit)




    
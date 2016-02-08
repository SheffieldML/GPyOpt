import numpy as np
import itertools

class Design_space(object):
    """
    The format of a input domain, possibly with restrictions:
    The domain is defined as a list of dictionaries contains a list of attributes, e.g.:

    - Arm bandit
    space  =[{'name': 'var_1', 'type': 'bandit', 'domain': [(-1,1),(1,0),(0,1)], 'dimensionality':2},
             {'name': 'var_2', 'type': 'bandit', 'domain': [(-1,4),(0,0),(1,2)]},

    - Continous domain
    space =[ {'name': 'var_1', 'type': 'continuous', 'domain':(-1,1), 'dimensionality':1},
             {'name': 'var_2', 'type': 'continuous', 'domain':(-3,1), 'dimensionality':2},
             {'name': 'var_3', 'type': 'bandit', 'domain': [(-1,1),(1,0),(0,1)], 'dimensionality':2},
             {'name': 'var_4', 'type': 'bandit', 'domain': [(-1,4),(0,0),(1,2)]},
             {'name': 'var_5', 'type': 'discrete', 'domain': (0,1,2,3)}]

    - Discrete domain
    space =[ {'name': 'var_3', 'type': 'discrete', 'domain': (0,1,2,3)}]
             {'name': 'var_3', 'type': 'discrete', 'domain': (-10,10)}]


    - Mixed domain 
    space =[{'name': 'var_1', 'type': 'continuous', 'domain':(-1,1), 'dimensionality':1},
            {'name': 'var_4', 'type': 'continuous', 'domain':(-3,1), 'dimensionality':2},
            {'name': 'var_3', 'type': 'discrete', 'domain': (0,1,2,3)}]

    Restrictions can be added to the problem. Each restriction is of the form c(x) <= 0 where c(x) is a function of 
    the input variables previously defined in the space. Restrictions should be written as a list
    of dictionaries. For instance, this would be an example of and space coupled with a constrain

    space =[ {'name': 'var_1', 'type': 'continuous', 'domain':(-1,1), 'dimensionality':2}]
    constrains = [ {'name': 'const_1', 'constrain': 'x[:,0]**2 + x[:,1]**2 - 1'}]

    If no constrains are provided the hypercube determined by the bounds constrains are used.

    """
    
    supported_types = ['continuous', 'discrete', 'bandit']
    
    def __init__(self, space, constrains=None):
        self._complete_attributes(space)
        self.constrains = constrains
        
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
                if d_out['domain'] == 'bandit': 
                    d_out['dimensionality'] = len(d_out['domain'][0])
                else:
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
    
    def get_continuous_space(self):
        space = []
        for d in self.space:
            if d['type']=='continuous':
                space += [d]
        return space

    def get_discrete_grid(self):
        sets_grid = []
        for d in self.space:
            if d['type']=='discrete':
                sets_grid.extend([d['domain']]*d['dimensionality'])
        return np.array(list(itertools.product(*sets_grid)))

    def get_bandit(self):
        arms_bandit = []
        for d in self.space:
            if d['type']=='bandit':
                arms_bandit += d['domain']
        return np.asarray(arms_bandit)

    def get_continuous_index(self):
        ## TODO
        index = []
        pass


    def indicator_constrains(self,x):
        x = np.atleast_2d(x)
        I_x = np.ones((x.shape[0],1))
        if self.constrains != None:
            for d in self.constrains:
                exec 'constrain =  lambda x:' + d['constrain']
                exec 'ind_x = (constrain(x)<0)*1'
                I_x *= ind_x.reshape(x.shape[0],1)
        return I_x


    
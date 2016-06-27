# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
import itertools

class Design_space(object):
    """
    Class to handle the input domain of the function. 
    The format of a input domain, possibly with restrictions:
    The domain is defined as a list of dictionaries contains a list of attributes, e.g.:

    - Arm bandit
    space  =[{'name': 'var_1', 'type': 'bandit', 'domain': [(-1,1),(1,0),(0,1)]},
             {'name': 'var_2', 'type': 'bandit', 'domain': [(-1,4),(0,0),(1,2)]}]

    - Continuous domain
    space =[ {'name': 'var_1', 'type': 'continuous', 'domain':(-1,1), 'dimensionality':1},
             {'name': 'var_2', 'type': 'continuous', 'domain':(-3,1), 'dimensionality':2},
             {'name': 'var_3', 'type': 'bandit', 'domain': [(-1,1),(1,0),(0,1)], 'dimensionality':2},
             {'name': 'var_4', 'type': 'bandit', 'domain': [(-1,4),(0,0),(1,2)]},
             {'name': 'var_5', 'type': 'discrete', 'domain': (0,1,2,3)}]

    - Discrete domain
    space =[ {'name': 'var_3', 'type': 'discrete', 'domain': (0,1,2,3)}]
             {'name': 'var_3', 'type': 'discrete', 'domain': (-10,10)}]


    - Mixed domain 
    space =[{'name': 'var_1', 'type': 'continuous', 'domain':(-1,1), 'dimensionality' :1},
            {'name': 'var_4', 'type': 'continuous', 'domain':(-3,1), 'dimensionality' :2},
            {'name': 'var_3', 'type': 'discrete', 'domain': (0,1,2,3)}]

    Restrictions can be added to the problem. Each restriction is of the form c(x) <= 0 where c(x) is a function of 
    the input variables previously defined in the space. Restrictions should be written as a list
    of dictionaries. For instance, this is an example of an space coupled with a constrain

    space =[ {'name': 'var_1', 'type': 'continuous', 'domain':(-1,1), 'dimensionality' :2}]
    constraints = [ {'name': 'const_1', 'constrain': 'x[:,0]**2 + x[:,1]**2 - 1'}]

    If no constrains are provided the hypercube determined by the bounds constraints are used.

    param space: list of dictionaries as indicated above.
    param constraints: list of dictionaries as indicated above (default, none)

    """
    
    supported_types = ['continuous', 'discrete', 'bandit']
    
    def __init__(self, space, constraints=None):
        self._complete_attributes(space)
        self.space_expanded = self._expand_attributes(self.space)
        self.constraints = constraints

        if self.has_types['bandit'] and (self.has_types['continuous'] or self.has_types['discrete']):
            print('Combinations of bandits with other variables are not supported.')

    @staticmethod
    def fromConfig(space, constraints):
        import six
        from ast import literal_eval

        for d in space:
            if isinstance(d['dimensionality'],six.string_types):
                d['dimensionality'] = int(d['dimensionality'])
            d['domain'] = literal_eval(d['domain'])
        return Design_space(space, None if len(constraints)==0 else constraints)
        
    def _complete_attributes(self, space):
        """
        Creates an internal dictionary where all the missing elements are completed.
        """   
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
                    dims = np.array([len (a) for a in d_out['domain']])
                    assert np.all(dims==dims[0]), 'The dimensionalities of the bandit variable '+d_out['name']+' have to be the same!'
                    d_out['dimensionality'] = len(d_out['domain'][0])
                else:
                    d_out['dimensionality'] = 1

            self.dimensionality += d_out['dimensionality']
            self.space.append(d_out)
            self.has_types[d_out['type']] = True

        dims = [d['dimensionality'] for d in self.space]
        self._var_indices = np.zeros((len(dims),2),dtype=np.int)
        offset = 0
        for i in range(len(dims)):
            self._var_indices[i,0] = offset
            self._var_indices[i,1] = offset + dims[i]
            offset += dims[i]

    def get_variable_indices(self):
        return self._var_indices

    def has_constrains(self):
        """
        Checks if the problem has constrains.
        """
        return self.constraints != None

    def _expand_attributes(self, space):
        """
        Creates and internal dictionary where the attributes with dimensionality larger than one are expanded. This
        dictionary is the one that is used internally to do the optimization.
        """
        space_expanded = []
        for d in space:
            d_new = []
            for i in range(d['dimensionality']):
                dd = d.copy()
                dd['dimensionality'] = 1
                dd['name'] = dd['name']+'_'+str(i+1)
                d_new += [dd] 
            space_expanded += d_new
        return space_expanded

    def get_continuous_bounds(self):
        """
        Extracts the bounds of the continuous variables.
        """
        bounds = []
        for d in self.space:
            if d['type']=='continuous':
                bounds.extend([d['domain']]*d['dimensionality'])
        return bounds
    
    def get_bounds(self):
        """
        Extracts the bounds of all the inputs of the domain.
        """
        bounds = []
        if self.has_types['bandit']:
            bandit = self.get_bandit()
            mins, maxs = bandit.min(axis=0).tolist(), bandit.max(axis=0).tolist()
            for i in range(len(bandit.min(axis=0).tolist())): bounds += [(mins[i],maxs[i])] 
        else:
            for d in self.space:
                if d['type']=='continuous':
                    bounds.extend([d['domain']]*d['dimensionality'])
                elif d['type']=='discrete':
                    bounds.extend([(min(d['domain'])-.1, max(d['domain'])+.1) ] *d['dimensionality'])
        return bounds

    def get_subspace(self,dims):
        '''
        Extracts a the subspace according to a list of dimension.
        '''
        subspace = []
        for i in dims:
            subspace += [self.space_expanded[i]]
        return subspace

    def get_continuous_space(self):
        """
        Extracts the list of dictionaries with continuous components
        """
        return [d for d in self.space if d['type']=='continuous']

    def get_discrete_grid(self):
        """
        Computes a Numpy array with the grid of points that results after crossing the possible outputs of the discrete 
        variables
        """
        sets_grid = []
        for d in self.space:
            if d['type']=='discrete':
                sets_grid.extend([d['domain']]*d['dimensionality'])
        return np.array(list(itertools.product(*sets_grid)))

    def get_bandit(self):
        """
        Extracts the arms of the bandit if any.
        """
        arms_bandit = []
        for d in self.space:
            if d['type']=='bandit':
                arms_bandit += tuple(map(tuple, d['domain']))
        return np.asarray(arms_bandit)

    def get_continuous_dims(self):
        """
        Returns the dimension of the continuous components of the domain.
        """
        continuous_dims = []
        for i in range(self.dimensionality):
            if self.space_expanded[i]['type']=='continuous':
                continuous_dims += [i]
        return continuous_dims

    def get_discrete_dims(self):
        """
        Returns the dimension of the discrete components of the domain.
        """
        discrete_dims = []
        for i in range(self.dimensionality):
            if self.space_expanded[i]['type']=='discrete':
                discrete_dims += [i]
        return discrete_dims

    def indicator_constraints(self,x):
        """
        Return zero if x is within the constrains and zero otherwise.
        """
        x = np.atleast_2d(x)
        I_x = np.ones((x.shape[0],1))
        if self.constraints != None:
            for d in self.constraints:
                try:
                    exec('constrain =  lambda x:' + d['constrain'],globals())
                    ind_x = (constrain(x)<0)*1
                    I_x *= ind_x.reshape(x.shape[0],1)
                except:
                    print('Fail to compile the constraint: '+str(d))
                    raise
        return I_x

    def input_dim(self):
        """
        Extracts the input dimension of the domain.
        """
        n_cont = len(self.get_continuous_dims())
        n_disc = len(self.get_discrete_dims())
        return n_cont + n_disc


def bounds_to_space(bounds):
    """
    Takes as input a list of tuples with bounds, and create a dictionary to be processed by the class Design_space. This function
    us used to keep the compatibility with previous versions of GPyOpt in which only bounded continuous optimization was possible
    (and the optimization domain passed as a list of tuples).
    """
    space = []
    for k in range(len(bounds)):
        space += [{'name': 'var_'+str(k+1), 'type': 'continuous', 'domain':bounds[k], 'dimensionality':1}]
    return space

    




    

# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
import itertools
from ...util.general import values_to_array, merge_values

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

    supported_types = ['continuous', 'discrete', 'bandit','categorical','context']

    def __init__(self, space, constraints=None):

        ## --- Complete and expand attributes
        self._complete_attributes(space)
        self.space_expanded = self._expand_attributes(self.space)
        self._compute_variable_indices()
        self._update_values_noncontinuous()

        ## -- Compute raw and model dimensionalities
        self.objective_dimensionality = len(self.space_expanded)
        self.model_input_dims = [d['dimensionality_in_model'] for d in self.space_expanded]
        self.model_dimensionality = sum(self.model_input_dims)

        ## -- Checking constraints
        self.constraints = constraints


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

        ### --- Complete attributes
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
            if 'parents' not in d_out:
                d['parents']= 'none'

            self.dimensionality += d_out['dimensionality']
            self.space.append(d_out)
            self.has_types[d_out['type']] = True


    def _expand_attributes(self, space):
        """
        Creates and internal dictionary where the attributes with dimensionality larger than one are expanded. This
        dictionary is the one that is used internally to do the optimization.
        """
        space_expanded = []

        ### -- Add terms with multiple dimensionalities
        for d in space:
            d_new = []
            for i in range(d['dimensionality']):
                dd = d.copy()
                dd['dimensionality'] = 1
                dd['name'] = dd['name']+'_'+str(i+1)
                d_new += [dd]
            space_expanded += d_new

        new_space_expanded = []

        ### --- Expand the bandits if any
        for d in space_expanded:
            if d['type']=='bandit':
                d_new = []
                for i in range(d['domain'].shape[1]):
                    dd = d.copy()
                    dd['name'] = dd['name']+'_'+str(i+1)
                    dd['domain'] = d['domain'][:,i]
                    d_new +=[dd]
                new_space_expanded +=d_new
            else:
                new_space_expanded +=[d]

        ### --- Add model dimensionalities an internal idx_discrete
        for d in new_space_expanded:
            ### --- Internal IDs
            d['idx'] = 'idx' + str(i)

            ### --- Model dimensionalities
            if d['type']=='categorical':
                d['dimensionality_in_model'] = len(d['domain'])
            else:
                d['dimensionality_in_model'] = 1

        return new_space_expanded


    def objective_to_model(self, x_objective):
        ''' This function serves as interface between objective input vectors and
        model input vectors'''

        x_model = np.empty((1,0))
        for k in range(self.objective_dimensionality):
            d = self.space_expanded[k]

            ### ---Handle separately the case for the discrete variables
            if d['type'] == 'categorical':
                new_entry = np.zeros((1,d['dimensionality_in_model']))
                new_entry[0,int(x_objective[0,k])] = 1
            else:
                new_entry = np.atleast_2d(x_objective[0,k])

            x_model = np.hstack((x_model,new_entry))
        return x_model

    def unzip_inptus(self,X):
        Z = np.empty((0,self.model_dimensionality))
        for k in range(X.shape[0]):
            Z = np.vstack((Z,self.objective_to_model(X[k,:][None,:])))
        return Z

    def zip_inputs(self,X):
        Z = np.empty((0,self.objective_dimensionality))
        for k in range(X.shape[0]):
            Z = np.vstack((Z,self.model_to_objective(X[k,:][None,:])))
        return Z

    def model_to_objective(self, x_model):
        ''' This function serves as interface between model input vectors and
            objective input vectors
        '''
        idx_model = 0
        x_objective = np.empty((1,0))

        for idx_obj in range(self.objective_dimensionality):
            d = self.space_expanded[idx_obj]
            vardim = d['dimensionality_in_model']
            original_entry = x_model[0,idx_model:(idx_model+vardim)]

            ### ---Handle separately the case for the discrete variables
            if d['type'] == 'categorical':
                new_entry = np.dot(np.atleast_2d(np.array(range(vardim))),np.atleast_2d(original_entry).T)
            else:
                new_entry = np.atleast_2d(x_model[0,idx_model])
            idx_model += vardim
            x_objective = np.hstack((x_objective,new_entry))

        return x_objective

    def has_constrains(self):
        """
        Checks if the problem has constrains. Note that the coordinates of the constrains are defined
        in terms of the model inputs and not in terms of the objective inputs. This means that if bandit or
        discre varaibles are in place, the restrictions should reflect this fact (TODO: implement the
        mapping of constraints defined on the objective to constrains defined on the model).
        """
        return self.constraints != None


    def get_bounds(self):
        """
        Extracts the bounds of all the inputs of the domain of the *model*
        """
        bounds = []

        for d in self.space_expanded:
            if d['type']=='continuous' or d['type']=='context' :
                bounds += [d['domain']]

            if d['type']=='bandit' or d['type']=='discrete':
                bounds += [(min(d['domain']),max(d['domain']))]

            if d['type'] == 'categorical':
                bounds += [(0,1)]*d['dimensionality_in_model']

        return bounds

    def _compute_variable_indices(self):
        '''
        Computes lists of indices of the different types of variables in the model.
        '''
        self.idx_continuous = []
        self.idx_context = []
        self.idx_discrete = []
        self.idx_categorical = []
        self.idx_bandits = []
        self.idx_noncontinuous = []

        k = 0
        for d in self.space_expanded:
            if d['type']=='continuous':
                self.idx_continuous += [k]
                k+= 1

            elif d['type']=='context' :
                self.idx_context += [k]
                k+= 1

            elif d['type']=='discrete':
                self.idx_discrete += [k]
                k+= 1

            elif d['type'] == 'categorical':
                self.idx_categorical += list(range(k,k+d['dimensionality_in_model']))
                k+=d['dimensionality_in_model']

            elif d['type']=='bandit':
                self.idx_bandits += [k]
                k+= 1

        self.idx_noncontinuous = list(np.sort(self.idx_context + self.idx_discrete + self.idx_categorical+ self.idx_bandits))

    def _update_values_noncontinuous(self):
        '''
        Computes a numpy array with all possible combinations that the non continuous
        variables can show
        '''
        self.values_noncontinuous = None

        for d in self.space:
            for dim in range(d['dimensionality']):
                ### --- Extract the new values
                if d['type'] == 'discrete':
                    new_values = d['domain']

                elif d['type'] == 'categorical':
                    new_values = np.eye(len(d['domain']))

                elif d['type'] == 'context':
                    new_values = d['value']

                elif d['type'] == 'bandit':
                    new_values = d['domain']

                ### --- Augment the possible values
                if d['type'] != 'continuous':
                    if self.values_noncontinuous == None:
                        self.values_noncontinuous = values_to_array(new_values)
                    else:
                        self.values_noncontinuous = merge_values(self.values_noncontinuous,new_values)


    def update_context(self,vals):
        vals = np.atleast_2d(np.array(vals))
        num_vals_assigned=0

        for d in self.space:
            if d['type'] == 'context':
                d['value'] = vals[0,num_vals_assigned]
                num_vals_assigned += 1
        if vals.shape[1] != num_vals_assigned:
            print("The number of context values must coincide with the number of contextual variables in the original space definition (withot the multipicity).")
        else:
            self.space_expanded = self._expand_attributes(self.space)
            self._update_values_noncontinuous()



    def get_subspace(self,dims):
        '''
        Extracts subspace from the reference of a list of variables in the inputs
        of the model.
        '''
        subspace = []
        k = 0
        for d in self.space_expanded:
            if k in dims:
                subspace += [d]
            if d['type']=='categorical':
                k += len(d['domain'])
            else:
                k +=1
        return subspace


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

#################### ------ ALL THE REAMINING FUNCIONS ARE REDUNDANT NOW AND SHOULD BE DEPRECATED


    ###
    ### ---- Atributes for the continuous variables
    ###

    def get_continuous_bounds(self):
        """
        Extracts the bounds of the continuous variables.
        """
        bounds = []
        for d in self.space:
            if d['type']=='continuous':
                bounds.extend([d['domain']]*d['dimensionality'])
        return bounds


    def get_continuous_dims(self):
        """
        Returns the dimension of the continuous components of the domain.
        """
        continuous_dims = []
        for i in range(self.dimensionality):
            if self.space_expanded[i]['type']=='continuous':
                continuous_dims += [i]
        return continuous_dims


    def get_continuous_space(self):
        """
        Extracts the list of dictionaries with continuous components
        """
        return [d for d in self.space if d['type']=='continuous']


    ###
    ### ---- Atributes for the discrete variables
    ###


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


    def get_discrete_dims(self):
        """
        Returns the dimension of the discrete components of the domain.
        """
        discrete_dims = []
        for i in range(self.dimensionality):
            if self.space_expanded[i]['type']=='discrete':
                discrete_dims += [i]
        return discrete_dims


    def get_discrete_space(self):
        """
        Extracts the list of dictionaries with continuous components
        """
        return [d for d in self.space if d['type']=='discrete']


    ###
    ### ---- Atributes for the bandits variables
    ###


    def get_bandit(self):
        """
        Extracts the arms of the bandit if any.
        """
        arms_bandit = []
        for d in self.space:
            if d['type']=='bandit':
                arms_bandit += tuple(map(tuple, d['domain']))
        return np.asarray(arms_bandit)







###
### ---- Other stuff, to mantain compatibility with previous versions.
###


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

# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
import itertools
from copy import deepcopy

from .variables import BanditVariable, DiscreteVariable, CategoricalVariable, ContinuousVariable, create_variable
from ..errors import InvalidConfigError
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
    of dictionaries. For instance, this is an example of an space coupled with a constraint

    space =[ {'name': 'var_1', 'type': 'continuous', 'domain':(-1,1), 'dimensionality' :2}]
    constraints = [ {'name': 'const_1', 'constraint': 'x[:,0]**2 + x[:,1]**2 - 1'}]

    If no constraints are provided the hypercube determined by the bounds constraints are used.

    Note about the internal representation of the vatiables: for variables in which the dimaensionality
    has been specified in the domain, a subindex is internally asigned. For instance if the variables
    is called 'var1' and has dimensionality 3, the first three positions in the internal representation
    of the domain will be occupied by variables 'var1_1', 'var1_2' and 'var1_3'. If no dimensionality
    is added, the internal naming remains the same. For instance, in the example above 'var3'
    should be fixed its original name.



    param space: list of dictionaries as indicated above.
    param constraints: list of dictionaries as indicated above (default, none)

    """

    supported_types = ['continuous', 'discrete', 'bandit','categorical']

    def __init__(self, space, constraints=None, store_noncontinuous = False):

        ## --- Complete and expand attributes
        self.store_noncontinuous = store_noncontinuous
        self.config_space = space

        ## --- Transform input config space into the objects used to run the optimization
        self._translate_space(self.config_space)
        self._expand_space()
        self._compute_variables_indices()
        self._create_variables_dic()

        ## -- Compute raw and model dimensionalities
        self.objective_dimensionality = len(self.space_expanded)
        self.model_input_dims = [v.dimensionality_in_model for v in self.space_expanded]
        self.model_dimensionality = sum(self.model_input_dims)

        # Because of the misspelling API used to expect "constrain" as a key
        # This fixes the API but also supports the old form
        if constraints is not None:
            for c in constraints:
                if 'constrain' in c:
                    c['constraint'] = c['constrain']
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

    def _expand_config_space(self):
        """
        Expands the config input space into a list of diccionaries, one for each variable_dic
        in which the dimensionality is always one.

        Example: It would transform
        config_space =[ {'name': 'var_1', 'type': 'continuous', 'domain':(-1,1), 'dimensionality':1},
                        {'name': 'var_2', 'type': 'continuous', 'domain':(-3,1), 'dimensionality':2},

        into

        config_expande_space =[ {'name': 'var_1', 'type': 'continuous', 'domain':(-1,1), 'dimensionality':1},
                      {'name': 'var_2', 'type': 'continuous', 'domain':(-3,1), 'dimensionality':1},
                      {'name': 'var_2_1', 'type': 'continuous', 'domain':(-3,1), 'dimensionality':1}]

        """
        self.config_space_expanded = []

        for variable in self.config_space:
            variable_dic = variable.copy()
            if 'dimensionality' in variable_dic.keys():
                dimensionality = variable_dic['dimensionality']
                variable_dic['dimensionality'] = 1
                variables_set = [variable_dic.copy() for d in range(dimensionality)]
                k=1
                for variable in variables_set:
                    variable['name'] = variable['name'] + '_'+str(k)
                    k+=1
                self.config_space_expanded += variables_set
            else:
                self.config_space_expanded += [variable_dic]

    def _compute_variables_indices(self):
        """
        Computes and saves the index location of each variable (as a list) in the objectives
        space and in the model space. If no categorical variables are available, these two are
        equivalent.
        """

        counter_objective = 0
        counter_model     = 0

        for variable in self.space_expanded:
            variable.set_index_in_objective([counter_objective])
            counter_objective +=1

            if variable.type is not 'categorical':
                variable.set_index_in_model([counter_model])
                counter_model +=1
            else:
                num_categories = len(variable.domain)
                variable.set_index_in_model(list(range(counter_model,counter_model + num_categories)))
                counter_model +=num_categories


    def find_variable(self,variable_name):
        if variable_name not in self.name_to_variable.keys():
            raise InvalidVariableNameError('Name of variable not in the input domain')
        else:
            return self.name_to_variable[variable_name]

    def _create_variables_dic(self):
        """
        Returns the variable by passing its name
        """
        self.name_to_variable = {}
        for variable in self.space_expanded:
            self.name_to_variable[variable.name] = variable

    def _translate_space(self, space):
        """
        Translates a list of dictionaries into internal list of variables
        """
        self.space = []
        self.dimensionality = 0
        self.has_types = d = {t: False for t in self.supported_types}

        for i, d in enumerate(space):
            descriptor = deepcopy(d)
            descriptor['name'] = descriptor.get('name', 'var_' + str(i))
            descriptor['type'] = descriptor.get('type', 'continuous')
            if 'domain' not in descriptor:
                raise InvalidConfigError('Domain attribute is missing for variable ' + descriptor['name'])
            variable = create_variable(descriptor)
            self.space.append(variable)
            self.dimensionality += variable.dimensionality
            self.has_types[variable.type] = True

        # Check if there are any bandit and non-bandit variables together in the space
        if any(v.is_bandit() for v in self.space) and any(not v.is_bandit() for v in self.space):
            raise InvalidConfigError('Invalid mixed domain configuration. Bandit variables cannot be mixed with other types.')

    def _expand_space(self):
        """
        Creates an internal list where the variables with dimensionality larger than one are expanded.
        This list is the one that is used internally to do the optimization.
        """

        ## --- Expand the config space
        self._expand_config_space()

        ## --- Expand the space
        self.space_expanded = []
        for variable in self.space:
            self.space_expanded += variable.expand()

    def objective_to_model(self, x_objective):
        ''' This function serves as interface between objective input vectors and
        model input vectors'''

        x_model = []

        for k in range(self.objective_dimensionality):
            variable = self.space_expanded[k]
            new_entry = variable.objective_to_model(x_objective[0,k])
            x_model += new_entry

        return x_model

    def unzip_inputs(self,X):
        if self._has_bandit():
            Z = X
        else:
            Z = []
            for k in range(X.shape[0]):
                Z.append(self.objective_to_model(X[k,:][None,:]))
        return np.atleast_2d(Z)

    def zip_inputs(self,X):
        if self._has_bandit():
            Z = X
        else:
            Z = []
            for k in range(X.shape[0]):
                Z.append(self.model_to_objective(X[k,:][None,:]))
        return np.atleast_2d(Z)

    def model_to_objective(self, x_model):
        ''' This function serves as interface between model input vectors and
            objective input vectors
        '''
        idx_model = 0
        x_objective = []

        for idx_obj in range(self.objective_dimensionality):
            variable = self.space_expanded[idx_obj]
            new_entry = variable.model_to_objective(x_model, idx_model)
            x_objective += new_entry
            idx_model += variable.dimensionality_in_model

        return x_objective

    def has_constraints(self):
        """
        Checks if the problem has constraints. Note that the coordinates of the constraints are defined
        in terms of the model inputs and not in terms of the objective inputs. This means that if bandit or
        discre varaibles are in place, the restrictions should reflect this fact (TODO: implement the
        mapping of constraints defined on the objective to constraints defined on the model).
        """
        return self.constraints is not None


    def get_bounds(self):
        """
        Extracts the bounds of all the inputs of the domain of the *model*
        """
        bounds = []

        for variable in self.space_expanded:
            bounds += variable.get_bounds()

        return bounds

    def has_continuous(self):
        """
        Returns `true` if the space contains at least one continuous variable, and `false` otherwise
        """
        return any(v.is_continuous() for v in self.space)

    def _has_bandit(self):
        return any(v.is_bandit() for v in self.space)

    def get_subspace(self, dims):
        '''
        Extracts subspace from the reference of a list of variables in the inputs
        of the model.
        '''
        subspace = []
        k = 0
        for variable in self.space_expanded:
            if k in dims:
                subspace.append(variable)
            k += variable.dimensionality_in_model
        return subspace


    def indicator_constraints(self,x):
        """
        Returns array of ones and zeros indicating if x is within the constraints
        """
        x = np.atleast_2d(x)
        I_x = np.ones((x.shape[0],1))
        if self.constraints is not None:
            for d in self.constraints:
                try:
                    exec('constraint = lambda x:' + d['constraint'], globals())
                    ind_x = (constraint(x)<0)*1
                    I_x *= ind_x.reshape(x.shape[0],1)
                except:
                    print('Fail to compile the constraint: ' + str(d))
                    raise
        return I_x

    def input_dim(self):
        """
        Extracts the input dimension of the domain.
        """
        n_cont = len(self.get_continuous_dims())
        n_disc = len(self.get_discrete_dims())
        return n_cont + n_disc

    def round_optimum(self, x):
        """
        Rounds some value x to a feasible value in the design space.
        x is expected to be a vector or an array with a single row
        """
        x = np.array(x)
        if not ((x.ndim == 1) or (x.ndim == 2 and x.shape[0] == 1)):
            raise ValueError("Unexpected dimentionality of x. Got {}, expected (1, N) or (N,)".format(x.ndim))

        if x.ndim == 2:
            x = x[0]

        x_rounded = []
        value_index = 0
        for variable in self.space_expanded:
            var_value = x[value_index : value_index + variable.dimensionality_in_model]
            var_value_rounded = variable.round(var_value)

            x_rounded.append(var_value_rounded)
            value_index += variable.dimensionality_in_model

        return np.atleast_2d(np.concatenate(x_rounded))


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
            if d.type == 'continuous':
                bounds.extend([d.domain]*d.dimensionality)
        return bounds


    def get_continuous_dims(self):
        """
        Returns the dimension of the continuous components of the domain.
        """
        continuous_dims = []
        for i in range(self.dimensionality):
            if self.space_expanded[i].type == 'continuous':
                continuous_dims += [i]
        return continuous_dims


    def get_continuous_space(self):
        """
        Extracts the list of dictionaries with continuous components
        """
        return [d for d in self.space if d.type == 'continuous']


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
            if d.type == 'discrete':
                sets_grid.extend([d.domain]*d.dimensionality)
        return np.array(list(itertools.product(*sets_grid)))


    def get_discrete_dims(self):
        """
        Returns the dimension of the discrete components of the domain.
        """
        discrete_dims = []
        for i in range(self.dimensionality):
            if self.space_expanded[i].type == 'discrete':
                discrete_dims += [i]
        return discrete_dims


    def get_discrete_space(self):
        """
        Extracts the list of dictionaries with continuous components
        """
        return [d for d in self.space if d.type == 'discrete']


    ###
    ### ---- Atributes for the bandits variables
    ###


    def get_bandit(self):
        """
        Extracts the arms of the bandit if any.
        """
        arms_bandit = []
        for d in self.space:
            if d.type == 'bandit':
                arms_bandit += tuple(map(tuple, d.domain))
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

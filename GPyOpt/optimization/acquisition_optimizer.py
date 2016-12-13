# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from .optimizer import select_optimizer
from ..util.general import multigrid, samples_multidimensional_uniform
import numpy as np
from ..core.task.space import Design_space


def AcquisitionOptimizer(space, optimizer='lbfgs', current_X = None, **kwargs):
    """
    Chooser for the type of acquisition optimizer to use. The decision is based on the type of input space and
    whether it contains discrete, continuous, bandit variables or a mix of them. If the problem is defined for
    a mix of discrete and continuous variables the optimization is done as follows:
        - All possible combinations of the values of the discrete variables are computed.
        - For each combination the problem is solved for the remaining continuous variables.
        - The arg min of all the sub-problems is taken.
    Note that this may be slow in cases with many discrete variables. In the bandits settings not optimization is
    carried out. Since the space is finite the argmin is computed.

    :param space: design space class from GPyOpt.
    :param optimizer: optimizer to use. Can be selected among:
        - 'lbfgs': L-BFGS.
        - 'DIRECT': Dividing Rectangles.
        - 'CMA': covariance matrix adaptation.
    """

    # ### --- case 1: pure bandits (including context)
    if (space.has_types['bandit'] or space.has_types['discrete'] or space.has_types['categorical'] or space.has_types['context']) and not space.has_types['continuous']:
        return BanditAcqOptimizer(space, current_X, **kwargs)

    ### --- case 2: pure continous (no context)
    elif space.has_types['continuous'] and not space.has_types['context'] and not space.has_types['discrete'] and not space.has_types['bandit'] and not space.has_types['categorical']:
        return ContAcqOptimizer(space, optimizer=optimizer, **kwargs)

    ### --- case 3: continuous + (discrete, categorical, bandits or contexts)
    else:
        return MixedAcqOptimizer(space, optimizer=optimizer, **kwargs)





class AcquOptimizer(object):
    """
    Base class for the optimizers of the acquisition functions.

    :param space: design space class from GPyOpt.
    """

    def __init__(self, space):
        self.space = space

    def optimize(self, f=None, df=None, f_df=None):
        """
        :param f: function to optimize.
        :param df: gradient of the function to optimize.
        :param f_df: returns both the function to optimize and its gradient.
        """
        return None, None

class ContAcqOptimizer(AcquOptimizer):
    """
    General class for acquisition optimizers defined in continuous domains

    :param space: design space class from GPyOpt.
    :param optimizer: optimizer to use. Can be selected among:
        - 'lbfgs': L-BFGS.
        - 'DIRECT': Dividing Rectangles.
        - 'CMA': covariance matrix adaptation.
    :param n_samples: number of initial points in which the acquisition is evaluated.
    :param fast: whether just a local optimizer should be run starting in the best location (default, True). If False a local search is performed
            for each point and the best of all is taken.
    :param ramdom: whether the initial samples are taken randomly (or in a grid if False). Note that, if False, n_samples represent the number
            of points user per dimension.
    :param search: whether to do local search or not.
    """


    def __init__(self, space, optimizer='lbfgs', n_samples=5000, fast=True, random=True, search=True, **kwargs):
        super(ContAcqOptimizer, self).__init__(space)
        self.n_samples          = n_samples
        self.fast               = fast
        self.random             = random
        self.search             = search
        self.optimizer_name     = optimizer
        self.kwargs             = kwargs
        self.all_dims           = list(range(space.model_dimensionality))
        self.unfix_dimensions()


    def unfix_dimensions(self):
        '''
        Restores the initial configuration after a dimension has been fixed (and initializes the defalt one)
        '''
        self.fixed_dims         = []
        self.fixed_values       = None
        self.free_dims          = self.all_dims
        self.subspace           = self.space
        self.optimizer          = select_optimizer(self.optimizer_name)(self.space, **self.kwargs)
        self.free_bounds        = self.space.get_bounds()
        self.generate_initial_points()


    def fix_dimensions(self, dims=None, values=None):
        '''
        Fix the values of some of the dimensions. Once this this done the optimization is carried out only across the not fixed dimensions.

        :param dims: list of the indexes of the dimensions to fix.
        :param values: list of the values at which the selected dimensions should be fixed.
        '''

        self.fixed_dims         = dims
        self.fixed_values       = np.atleast_2d(values)
        self.free_dims          = [item for item in self.all_dims if item not in self.fixed_dims]

        ###
        self.subspace           = Design_space(self.space.get_subspace(self.free_dims))


        self.optimizer          = select_optimizer(self.optimizer_name)(self.subspace, **self.kwargs)
        self.free_bounds        = self.subspace.get_bounds()
        self.generate_initial_points()



    def generate_initial_points(self):
        if self.random:
            self.samples = samples_multidimensional_uniform(self.free_bounds,self.n_samples)
        else:
            self.samples = multigrid(self.free_bounds, self.n_samples)


    def _expand_vector(self,x):
        '''
        Takes a value x in the subspace of not fixed dimensions and expands it with the values of the fixed ones.
        '''
        xx = np.zeros((x.shape[0],self.space.model_dimensionality))
        xx[:,np.array(self.free_dims)]  = x
        if self.space.dimensionality != len(self.free_dims):
            xx[:,np.array(self.fixed_dims)] = self.fixed_values
        return xx

    def optimize(self, f=None, df=None, f_df=None):
        """
        Optimizes the input function.

        :param f: function to optimize.
        :param df: gradient of the function to optimize.
        :param f_df: returns both the function to optimize and its gradient.

        """
        self.f = f
        self.df = df
        self.f_df = f_df

        def fp(x):
            '''
            Wrapper of *f*: takes an input x with size of the not fixed dimensions expands it and evaluates the entire function.
            '''
            x = np.atleast_2d(x)
            xx = self._expand_vector(x)
            if x.shape[0]==1:
                return self.f(xx)[0]
            else:
                return self.f(xx)

        def fp_dfp(x):
            '''
            Wrapper of the derivative of *f*: takes an input x with size of the not fixed dimensions expands it and evaluates the gradient of the entire function.
            '''
            x = np.atleast_2d(x)
            xx = self._expand_vector(x)

            fp_xx , dfp_xx = f_df(xx)
            dfp_xx = dfp_xx[:,np.array(self.free_dims)]
            return fp_xx, dfp_xx

        ## --- The optimization is done here

        ## --- Fast method: only runs a local optimizer at the best found evaluation
        if self.fast:
            pred_fp = fp(self.samples)
            x0 =  self.samples[np.argmin(pred_fp)]
            if self.search:
                if self.f_df == None: fp_dfp = None  # -- In case no gradients are available
                x_min, f_min = self.optimizer.optimize(x0, f =fp, df=None, f_df=fp_dfp)
                return self._expand_vector(x_min), f_min
            else:
                return self._expand_vector(np.atleast_2d(x0)), pred_fp
        else:
        ## --- Standard method: runs a local optimizer at all the acquisition evaluation
            x_min = None
            f_min = np.Inf
            for i in self.samples.shape[0]:
                if self.search:
                    if self.f_df == None: fp_dfp = None # -- In case no gradients are available
                    x1, f1 = self.optimizer.optimize(self.samples[i], f =fp, df=None, f_df=fp_dfp)
                else:
                    x1, f1 = self.samples[i], fp(self.samples[i])
                if f1<f_min:
                    x_min = x1
                    f_min = f1
            return self._expand_vector(x_min), f_min






class MixedAcqOptimizer(AcquOptimizer):
    """
    General class for acquisition optimizers defined on mixed domains (continuous + other variables
    supported by GPyOpt)

    :param space: design space class from GPyOpt.
    :param optimizer: optimizer to use. Can be selected among:
        - 'lbfgs': L-BFGS.
        - 'DIRECT': Dividing Rectangles.
        - 'CMA': covariance matrix adaptation.
    :param n_samples: number of initial points in which the acquisition is evaluated.
    :param fast: whether just a local optimizer should be run starting in the best location (default, True). If False a local search is performed
            for each point and the best of all is taken.
    :param ramdom: whether the initial samples are taken randomly (or in a grid if False). Note that, if False, n_samples represent the number
            of points user per dimension.
    :param search: whether to do local search or not.
    """

    def __init__(self, space, optimizer='lbfgs', n_samples=1000, fast=True, random=True, search=True, **kwargs):
        super(MixedAcqOptimizer, self).__init__(space)

        self.space = space
        self.mixed_optimizer = ContAcqOptimizer(space, n_samples=n_samples, fast=fast, random=random, search=search, optimizer=optimizer, **kwargs)
        self.idx_noncontinuous = self.space.idx_noncontinuous


    def _update_values_noncontinuous(self):
        '''
        Updates the values of the non continuous variables according to the current
        values stored in the space.
        '''
        self.values_noncontinuous = self.space.values_noncontinuous


    def optimize(self, f=None, df=None, f_df=None):
        """
        Optimization of the acquisition. This function otimizes the continuous variables for all possible
        combination of the discrete, bandits and categorical.

        :param f: function to optimize.
        :param df: gradient of the function to optimize.
        :param f_df: returns both the function to optimize and its gradient.
        """

        self._update_values_noncontinuous()
        num_noncontinuous = self.values_noncontinuous.shape[0]
        partial_x_min  = np.zeros((num_noncontinuous,self.space.model_dimensionality))
        partial_f_min  = np.zeros((num_noncontinuous,1))

        ### --- Loop accros the combinations
        for i in range(num_noncontinuous):
            self.mixed_optimizer.fix_dimensions(dims=self.idx_noncontinuous, values=self.values_noncontinuous[i,:])
            partial_x_min[i,:] , partial_f_min[i,:] = self.mixed_optimizer.optimize(f, df, f_df)
            self.mixed_optimizer.unfix_dimensions()

        return np.atleast_2d(partial_x_min[np.argmin(partial_f_min)]), np.atleast_2d(min(partial_f_min))


class BanditAcqOptimizer(AcquOptimizer):
    """
    General class for acquisition optimizers defined on bandits

    :param space: design space class from GPyOpt.
    :param current_X: numpy array containing the arms of the bandit that hasn't been pulled yet.
    """

    def __init__(self, space, current_X, **kwargs):
        super(BanditAcqOptimizer, self).__init__(space)
        self.space = space
        self.pulled_arms = current_X

    def optimize(self, f=None, df=None, f_df=None):
        """
        Optimization of the acquisition. Since it is a bandit it just takes the argmin of the outputs.

        :param f: function to optimize.
        :param df: gradient of the function to optimize.
        :param f_df: returns both the function to optimize and its gradient.
        """

        # --- Get all potential arms
        if self.space.has_types['discrete']:
            arms = self.space.get_discrete_grid()
        else:
            arms = self.space.get_bandit()

        if arms.shape[0] > self.pulled_arms.shape[0]:
            # --- remove select best arm not yet sampled
            pref_f = f(arms)
            index = np.argsort(pref_f.flatten())
            k=0
            while any((self.pulled_arms[:]==arms[index[k],:].flatten()).all(1)):
                k +=1

            x_min = arms[index[k],:]
            f_min = f(x_min)
            #self.pulled_arms = np.vstack((self.pulled_arms, x_min))

        else:
            import sys
            sys.exit('All locations of the design space have been sampled.')

        return np.atleast_2d(x_min), f_min

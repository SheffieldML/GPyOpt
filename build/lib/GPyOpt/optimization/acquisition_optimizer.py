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
    
    if space.has_types['bandit'] and (space.has_types['continuous'] or space.has_types['discrete']):
        raise Exception('Not possible to combine bandits with other variable types.)')

    elif space.has_types['bandit']:
        return BanditAcqOptimizer(space, current_X, **kwargs)

    elif space.has_types['continuous'] and not space.has_types['discrete']:
        return ContAcqOptimizer(space, optimizer=optimizer, **kwargs)

    elif space.has_types['continuous'] and  space.has_types['discrete']:
        return MixedAcqOptimizer(space, optimizer=optimizer, **kwargs)

    elif not space.has_types['continuous'] and space.has_types['discrete']:
        return BanditAcqOptimizer(space, current_X, **kwargs)


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
        
        self.n_samples = n_samples
        self.fast= fast
        self.random = random
        self.search = search
        self.optimizer_name = optimizer
        self.kwargs = kwargs
        self.optimizer = select_optimizer(self.optimizer_name)(space, **kwargs)
        self.free_dims = list(range(space.dimensionality))
        self.bounds = self.space.get_bounds()
        self.subspace = self.space

        if self.random:
            self.samples = samples_multidimensional_uniform(self.bounds,self.n_samples)
        else:
            self.samples = multigrid(self.bounds, self.n_samples)


    def fix_dimensions(self, dims=None, values=None):
        '''
        Fix the values of some of the dimensions. Once this this done the optimization is carried out only across the not fixed dimensions.

        :param dims: list of the indexes of the dimensions to fix.
        :param values: list of the values at which the selected dimensions should be fixed.
        ''' 
        self.fixed_dims = dims
        self.fixed_values = np.atleast_2d(values)
        
        # -- restore to initial values
        self.free_dims = list(range(self.space.dimensionality))
        self.bounds = self.space.get_bounds()

        # -- change free dimensions and remove bounds from fixed dimensions
        for idx in self.fixed_dims[::-1]: # need to reverse the order to start removing from the back, otherwise dimensions dont' match
            self.free_dims.remove(idx)
            del self.bounds[idx]

        # -- take only the fixed components of the random samples
        self.samples = self.samples[:,np.array(self.free_dims)] # take only the component of active dims
        self.subspace = self.space.get_subspace(self.free_dims)
        self.optimizer = select_optimizer(self.optimizer_name)(Design_space(self.subspace), **self.kwargs)

    def _expand_vector(self,x):
        '''
        Takes a value x in the subspace of not fixed dimensions and expands it with the values of the fixed ones.
        '''
        xx = np.zeros((x.shape[0],self.space.dimensionality)) 
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
            
        else:
            print('All locations of the design space have been sampled.')
            #break
        
        self.pulled_arms = np.vstack((self.pulled_arms, x_min))


        # --- Previous approach: do not remove those already sampled
        # pref_f = f(arms)
        # x_min = arms[np.argmin(pref_f)]
        # f_min = f(x_min)

        return np.atleast_2d(x_min), f_min


class MixedAcqOptimizer(AcquOptimizer):
    """
    General class for acquisition optimizers defined on mixed domains of continuous and discrete variables. 

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
        self.discrete_dims = self.space.get_discrete_dims()
        self.discrete_values = self.space.get_discrete_grid()

    def optimize(self, f=None, df=None, f_df=None):
        """
        Optimization of the acquisition. 

        :param f: function to optimize.
        :param df: gradient of the function to optimize.
        :param f_df: returns both the function to optimize and its gradient.
        """
        num_discrete = self.discrete_values.shape[0]
        partial_x_min  = np.zeros((num_discrete,self.space.dimensionality))
        partial_f_min  = np.zeros((num_discrete,1))
        
        for i in range(num_discrete):
            self.mixed_optimizer.fix_dimensions(dims=self.discrete_dims, values=self.discrete_values[i,:])
            partial_x_min[i,:] , partial_f_min[i,:] = self.mixed_optimizer.optimize(f, df, f_df)

        return np.atleast_2d(partial_x_min[np.argmin(partial_f_min)]), np.atleast_2d(min(partial_f_min))





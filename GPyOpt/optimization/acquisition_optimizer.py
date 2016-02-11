
from .optimizer import select_optimizer
from ..util.general import multigrid, samples_multidimensional_uniform
import numpy as np
from ..core.task.space import Design_space


def AcquisitionOptimizer(space, n_samples=2000, fast=True, random=True, search=True, optimizer='lbfgs', **kwargs):
    
    if space.has_types['bandit'] and (space.has_types['continuous'] or space.has_types['discrete']):
        raise Exception('Not possible to combine bandits with other variable types.)')

    elif space.has_types['bandit']:
        return BanditAcqOptimizer(space, **kwargs)

    elif space.has_types['continuous'] and not space.has_types['discrete']:
        return ContAcqOptimizer(space, n_samples=n_samples, fast=fast, random=random, search=search, optimizer=optimizer, **kwargs)

    elif space.has_types['continuous'] and  space.has_types['discrete']:
        return MixedAcqOptimizer(space, n_samples=n_samples, fast=fast, random=random, search=search, optimizer=optimizer, **kwargs)

    elif not space.has_types['continuous'] and space.has_types['discrete']:
        return BanditAcqOptimizer(space, **kwargs)


class AcquOptimizer(object):
    
    def __init__(self, space):
        self.space = space
        
    def optimize(self, f=None, df=None, f_df=None):
         return None, None

class ContAcqOptimizer(AcquOptimizer):
    
    def __init__(self, space, n_samples=5, fast=True, random=True, search=True, optimizer='lbfgs', **kwargs):
        super(ContAcqOptimizer, self).__init__(space)
        
        self.n_samples = n_samples
        self.fast= fast
        self.random = random
        self.search = search
        self.optimizer_name = optimizer
        self.kwargs = kwargs
        self.optimizer = select_optimizer(self.optimizer_name)(space, **kwargs)
        self.free_dims = range(space.dimensionality)
        self.bounds = self.space.get_bounds()
        self.subspace = self.space

        if self.random:
            self.samples = samples_multidimensional_uniform(self.bounds,self.n_samples)
        else:
            self.samples = multigrid(self.bounds, self.n_samples)


    def fix_dimensions(self, dims=None, values=None):
        '''
        Fix the values of some of the dimensions
        ''' 
        self.fixed_dims = dims
        self.fixed_values = np.atleast_2d(values)
        
        # -- restore to initial values
        self.free_dims = range(self.space.dimensionality) 
        self.bounds = self.space.get_bounds()

        # -- change free dimensions and remove bounds from fixed dimensions
        for idx in self.fixed_dims: 
            self.free_dims.remove(idx)
            del self.bounds[idx]

        # -- take only the fixed components of the random samples
        self.samples = self.samples[:,np.array(self.free_dims)] # take only the component of active dims
        self.subspace = self.space.get_subspace(self.free_dims)
        self.optimizer = select_optimizer(self.optimizer_name)(Design_space(self.subspace), self.kwargs)

    def _expand_vector(self,x):
        '''
        Takes a value x in the subspace and expands it with the fixed values
        '''
        xx = np.zeros((x.shape[0],self.space.dimensionality)) 
        xx[:,np.array(self.free_dims)]  = x  
        if self.space.dimensionality != len(self.free_dims):
            xx[:,np.array(self.fixed_dims)] = self.fixed_values
        return xx

    def optimize(self, f=None, df=None, f_df=None):
        self.f = f
        self.df = df
        self.f_df = f_df

        def fp(x):  # evaluation of the function with some fixed dimensions
            '''
            x has dimesion of the free indices
            '''
            x = np.atleast_2d(x)
            xx = self._expand_vector(x)        
            if x.shape[0]==1:
                return self.f(xx)[0]
            else:
                return self.f(xx)

        def fp_dfp(x):  # evaluation and gradient of the function with some fixed dimensions
            '''
            x has dimesion of the free indices
            '''
            x = np.atleast_2d(x)
            xx = self._expand_vector(x)        
            
            fp_xx , dfp_xx = f_df(xx)
            dfp_xx = dfp_xx[:,np.array(self.free_dims)]
            return fp_xx, dfp_xx

        if self.fast:
            pred_fp = fp(self.samples)
            x0 =  self.samples[np.argmin(pred_fp)]
            if self.search:
                x_min, f_min = self.optimizer.optimize(x0, f =fp, df=None, f_df=fp_dfp)
                return self._expand_vector(x_min), f_min
            else:
                return self._expand_vector(np.atleast_2d(x0)), pred_fp
        else:
            x_min = None
            f_min = np.Inf
            for i in self.samples.shape[0]:
                if self.search:
                    x1, f1 = self.optimizer.optimize(self.samples[i], f =fp, df=None, f_df=fp_dfp)
                else:
                    x1, f1 = self.samples[i], fp(self.samples[i])
                if f1<f_min:
                    x_min = x1
                    f_min = f1
            return self._expand_vector(x_min), f_min
        

class BanditAcqOptimizer(AcquOptimizer):

    def __init__(self, space, **kwars):
        super(BanditAcqOptimizer, self).__init__(space)

    def optimize(self, f=None, df=None, f_df=None):
        pref_f = f(self.space.get_bandit())
        x_min = self.space.get_bandits()[np.argmin(pref_f)]
        f_min = f(x_min)
        return x_min, f_min


class MixedAcqOptimizer(AcquOptimizer):

    def __init__(self, space, n_samples, fast=True, random=True, search=True, optimizer='lbfgs', **kwargs):
        super(MixedAcqOptimizer, self).__init__(space)

        self.space = space
        self.mixed_optimizer = ContAcqOptimizer(space, n_samples=n_samples, fast=fast, random=random, search=search, optimizer=optimizer, **kwargs)
        self.discrete_dims = self.space.get_discrete_dims()
        self.discrete_values = self.space.get_discrete_grid()

    def optimize(self, f=None, df=None, f_df=None):
        num_discrete = self.discrete_values.shape[0]
        partial_x_min  = np.zeros((num_discrete,self.space.dimensionality))
        partial_f_min  = np.zeros((num_discrete,1))
        
        for i in range(num_discrete ):
            self.mixed_optimizer.fix_dimensions(dims=self.discrete_dims, values=self.discrete_values[i,:])
            partial_x_min[i,:] , partial_f_min[i,:] = self.mixed_optimizer.optimize(f, df, f_df)

        return np.atleast_2d(partial_x_min[np.argmin(partial_f_min)]), np.atleast_2d(min(partial_f_min))






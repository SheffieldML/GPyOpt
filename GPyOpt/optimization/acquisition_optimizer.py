
from .optimizer import select_optimizer
from ..util.general import multigrid, samples_multidimensional_uniform
import numpy as np

class AcquisitionOptimizer(object):
    
    def __init__(self, space):
        self.space = space
        
    def optimize(self, f=None, df=None, f_df=None):
        return None, None


class ContAcqOptimizer(AcquisitionOptimizer):
    
    def __init__(self, space, n_samples=5, fast=True, random=True, search=True, optimizer='lbfgs', **kw):
        super(ContAcqOptimizer, self).__init__(space)
        self.n_samples = n_samples
        self.fast= fast
        self.random = random
        self.search = search
        self.optimizer = select_optimizer(optimizer)(space, **kw)
        
        # draw_samples
        if self.random:
            self.samples = samples_multidimensional_uniform(self.space.get_continuous_bounds(),self.n_samples)
        else:
            self.samples = multigrid(self.space.get_continuous_bounds(), self.n_samples)
        
    def optimize(self, f=None, df=None, f_df=None):
        if self.fast:
            pred_f = f(self.samples)
            x0 =  self.samples[np.argmin(pred_f)]
            if self.search:
                return self.optimizer.optimize(x0, f, df, f_df)
            else:
                return np.atleast_2d(x0), pred_f
        else:
            x_min = None
            f_min = np.Inf
            for i in self.samples.shape[0]:
                if self.search:
                    x1, f1 = self.optimizer.optimize(self.samples[i], f, df, f_df)
                else:
                    x1, f1 = self.samples[i], f(self.samples[i])
                if f1<f_min:
                    x_min = x1
                    f_min = f1
            return x_min, f_min
        

class BanditAcqOptimizer(AcquisitionOptimizer):

    def __init__(self, space, **kw):
        super(BanditAcqOptimizer, self).__init__(space)

    def optimize(self, f=None, df=None, f_df=None):
        pref_f = f(self.space.get_bandit())
        x_min = self.space.get_bandits()[np.argmin(pref_f)]
        f_min = f(x_min)
        return x_min, f_min


class MixedAcqOptimizer(AcquisitionOptimizer):

    def __init__(self, space, n_samples, fast=True, random=True, search=True, optimizer='lbfgs', **kw):
        super(MixedAcqOptimizer, self).__init__(space)
        self.n_samples = n_samples
        self.fast= fast
        self.random = random
        self.search = search
        self.optimizer = select_optimizer(optimizer)(space, **kw)
        
        # draw_samples
        if self.random:
            self.samples = samples_multidimensional_uniform(self.space.get_continuous_bounds(),self.n_samples)
        else:
            self.samples = multigrid(self.space.get_continuous_bounds(), self.n_samples)

        self.continuous_space_optimizer = ContAcqOptimizer(self.get_continuous_space(),n_samples, fast, random, search, optimizer, kw)

    def optimize(self, f=None, df=None, f_df=None):
        
        # discrete_points = self.space.get_discrete_grid()
        # n_points = discrete_points.shape[0]
        # discrete_optima = np.zeros((n_points,1))
        # index = self.space.get_continuous_index():

        # for i in range(n_points):
        #     partial = partial_evaluator(f,df,f_df,index,discrete_points[i,:])
        #     discrete_optima[i,:],_ =  self.continuous_space_optimizer.optimize(partial.f,partial.df,partial.f_df)

        # # take the min
        pass

class partial_evaluator(object):
    '''
    Class that wraps a function and its derivative and enables to fix some components
    '''
    def __init__(self,f,df,f_df,index,values_index):
        self.f = f
        self.df = df
        self.f_df = f_df
        self.index = index
        self.values_index = values_index
    
    def f(self,x):
        return self.f(self._fix_entries(x)).reshape((x.shape[0],1))
    
    def df(self,x):
        return self.df(self._fix_entries(x))
    
    def f_df(self,x):
        return self.f_df(self._fix_entries(x))

    def _fix_entries(self,x):
        x[:,np.array(self.index)] = np.dot(np.ones((x.shape[0],1)),self.values_index)
        return x






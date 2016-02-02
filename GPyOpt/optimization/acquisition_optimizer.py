
from .optimizer import select_optimizer
from ..util.general import multigrid, samples_multidimensional_uniform
import numpy as np

class AcquisitionOptimizer(object):
    
    def __init__(self, space):
        self.space = space
        
    def optimize(self, f=None, df=None, f_df=None):
        return None, None


class ContAcqOptimizer(AcquisitionOptimizer):
    
    def __init__(self, space, n_samples, fast=True, random=True, search=True, optimizer='lbfgs', **kw):
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
                return x0, pred_f
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
        


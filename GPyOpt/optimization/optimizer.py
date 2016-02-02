
class Optimizer(object):
    
    def __init__(self, space):
        self.space = space
        
    def optimize(self, x0, f=None, df=None, f_df=None):
        return None, None
    
class Opt_lbfgs(Optimizer):
    def __init__(self, space, maxiter=1000):
        super(Opt_lbfgs, self).__init__(space)
        self.maxiter = maxiter
        assert self.space.has_types['continuous']
        
    def optimize(self, x0, f=None, df=None, f_df=None):
        import scipy.optimize
        if f_df is None: f_df = lambda x: (float(f(x)), df(x))
        if f_df is None and df is None:
            res = scipy.optimize.fmin_l_bfgs_b(f, x0=x0, bounds=self.space.get_continuous_bounds(),approx_grad=True, maxiter=self.maxiter)
        else:
            res = scipy.optimize.fmin_l_bfgs_b(f_df, x0=x0, bounds=self.space.get_continuous_bounds(), maxiter=self.maxiter)
        return res[0],res[1]
        
def select_optimizer(name):
    if name=='lbfgs':
        return Opt_lbfgs
        
        
        
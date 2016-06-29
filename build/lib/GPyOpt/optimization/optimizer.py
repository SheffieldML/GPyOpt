# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

def select_optimizer(name):
    """
    Chooser between different optimizers
    """

    if name == 'lbfgs':
        return Opt_lbfgs
    elif name == 'DIRECT':
        return Opt_DIRECT
    elif name == 'CMA':
        return Opt_CMA
    else:
        raise Exception('Invalid optimizer selected.')

class Optimizer(object):
    """
    Class for a general acquisition optimizer.

    :param space: design space GPyOpt class.
    """
    
    def __init__(self, space):
        self.space = space
        
    def optimize(self, x0, f=None, df=None, f_df=None):
        """
        :param x0: initial point for a local optimizer.
        :param f: function to optimize.
        :param df: gradient of the function to optimize.
        :param f_df: returns both the function to optimize and its gradient.
        """
        return None, None
    

class Opt_lbfgs(Optimizer):
    '''
    Wrapper for l-bfgs-b to use the true or the approximate gradients. 
    '''
    def __init__(self, space, maxiter=1000):
        super(Opt_lbfgs, self).__init__(space)
        self.maxiter = maxiter
        assert self.space.has_types['continuous']
        
    def optimize(self, x0, f=None, df=None, f_df=None):
        """
        :param x0: initial point for a local optimizer.
        :param f: function to optimize.
        :param df: gradient of the function to optimize.
        :param f_df: returns both the function to optimize and its gradient.
        """

        import scipy.optimize
        import numpy as np
        if f_df is None and df is not None: f_df = lambda x: float(f(x)), df(x)
        if f_df is not None:
            def _f_df(x):
                return f(x), f_df(x)[1][0]
        if f_df is None and df is None:
            res = scipy.optimize.fmin_l_bfgs_b(f, x0=x0, bounds=self.space.get_bounds(),approx_grad=True, maxiter=self.maxiter)
        else:
            res = scipy.optimize.fmin_l_bfgs_b(_f_df, x0=x0, bounds=self.space.get_bounds(), maxiter=self.maxiter)
        return np.atleast_2d(res[0]),np.atleast_2d(res[1])



class Opt_DIRECT(Optimizer):
    '''
    Wrapper for DIRECT optimization method. It works partitioning iteratively the domain 
    of the function. Only requires f and the box constrains to work.

    '''
    def __init__(self, space, maxiter=1000):
        super(Opt_DIRECT, self).__init__(space)
        self.maxiter = maxiter
        assert self.space.has_types['continuous']

    def optimize(self, x0=None, f=None, df=None, f_df=None):
        """
        :param x0: initial point for a local optimizer.
        :param f: function to optimize.
        :param df: gradient of the function to optimize.
        :param f_df: returns both the function to optimize and its gradient.
        """
        try:
            from DIRECT import solve
            import numpy as np
            def DIRECT_f_wrapper(f):
                def g(x, user_data):
                    return f(np.array([x])), 0
                return g
            lB = np.asarray(self.space.get_bounds())[:,0]
            uB = np.asarray(self.space.get_bounds())[:,1]
            x,_,_ = solve(DIRECT_f_wrapper(f),lB,uB, maxT=self.maxiter)
            return np.atleast_2d(x), f(np.atleast_2d(x))
        except:
            print("Cannot find DIRECT library, please install it to use this option.")


class Opt_CMA(Optimizer):
    '''
    Wrapper the Covariance Matrix Adaptation Evolutionary strategy (CMA-ES) optimization method. It works generating 
    an stochastic search based on multivariate Gaussian samples. Only requires f and the box constrains to work.

    '''
    def __init__(self, space, maxiter=1000):
        super(Opt_CMA, self).__init__(space)
        self.maxiter = maxiter
        assert self.space.has_types['continuous']
        try:
            import cma 
            import numpy as np
            def CMA_f_wrapper(f):
                def g(x):
                    return f(np.array([x]))[0][0]
                return g
            lB = np.asarray(self.space.get_bounds())[:,0]
            uB = np.asarray(self.space.get_bounds())[:,1]
            x = cma.fmin(CMA_f_wrapper(f), (uB + lB) * 0.5, 0.6, options={"bounds":[lB, uB], "verbose":-1})[0]
            print(x)
            return np.atleast_2d(x), f(np.atleast_2d(x))
        except:
            print("Cannot find cma library, please install it to use this option.")







        
        
        

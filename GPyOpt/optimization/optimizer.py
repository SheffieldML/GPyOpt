# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from ..core.errors import InvalidVariableNameError


class Optimizer(object):
    """
    Class for a general acquisition optimizer.

    :param bounds: list of tuple with bounds of the optimizer
    """

    def __init__(self, bounds):
        self.bounds = bounds

    def optimize(self, x0, f=None, df=None, f_df=None):
        """
        :param x0: initial point for a local optimizer.
        :param f: function to optimize.
        :param df: gradient of the function to optimize.
        :param f_df: returns both the function to optimize and its gradient.
        """
        raise NotImplementedError("The optimize method is not implemented in the parent class.")


class OptLbfgs(Optimizer):
    '''
    Wrapper for l-bfgs-b to use the true or the approximate gradients.
    '''
    def __init__(self, bounds, maxiter=1000):
        super(OptLbfgs, self).__init__(bounds)
        self.maxiter = maxiter

    def optimize(self, x0, f=None, df=None, f_df=None):
        """
        :param x0: initial point for a local optimizer.
        :param f: function to optimize.
        :param df: gradient of the function to optimize.
        :param f_df: returns both the function to optimize and its gradient.
        """
        import scipy.optimize
        if f_df is None and df is not None: f_df = lambda x: float(f(x)), df(x)
        if f_df is not None:
            def _f_df(x):
                return f(x), f_df(x)[1][0]
        if f_df is None and df is None:
            res = scipy.optimize.fmin_l_bfgs_b(f, x0=x0, bounds=self.bounds,approx_grad=True, maxiter=self.maxiter)
        else:
            res = scipy.optimize.fmin_l_bfgs_b(_f_df, x0=x0, bounds=self.bounds, maxiter=self.maxiter)

        ### --- We check here if the the optimizer moved. It it didn't we report x0 and f(x0) as scipy can return NaNs
        if res[2]['task'] == b'ABNORMAL_TERMINATION_IN_LNSRCH':
            result_x  = np.atleast_2d(x0)
            result_fx =  np.atleast_2d(f(x0))
        else:
            result_x = np.atleast_2d(res[0])
            result_fx = np.atleast_2d(res[1])

        return result_x, result_fx


class OptDirect(Optimizer):
    '''
    Wrapper for DIRECT optimization method. It works partitioning iteratively the domain
    of the function. Only requires f and the box constraints to work.

    '''
    def __init__(self, bounds, maxiter=1000):
        super(OptDirect, self).__init__(bounds)
        self.maxiter = maxiter
        assert self.space.has_types['continuous']

    def optimize(self, x0, f=None, df=None, f_df=None):
        """
        :param x0: initial point for a local optimizer.
        :param f: function to optimize.
        :param df: gradient of the function to optimize.
        :param f_df: returns both the function to optimize and its gradient.
        """
        # Based on the documentation of DIRECT, it does not seem we can pass through an initial point x0
        try:
            from DIRECT import solve
            def DIRECT_f_wrapper(f):
                def g(x, user_data):
                    return f(np.array([x])), 0
                return g
            lB = np.asarray(self.bounds)[:,0]
            uB = np.asarray(self.bounds)[:,1]
            x,_,_ = solve(DIRECT_f_wrapper(f),lB,uB, maxT=self.maxiter)
            return np.atleast_2d(x), f(np.atleast_2d(x))
        except ImportError:
            print("Cannot find DIRECT library, please install it to use this option.")


class OptCma(Optimizer):
    '''
    Wrapper the Covariance Matrix Adaptation Evolutionary strategy (CMA-ES) optimization method. It works generating
    an stochastic search based on multivariate Gaussian samples. Only requires f and the box constraints to work.

    '''
    def __init__(self, bounds, maxiter=1000):
        super(OptCma, self).__init__(bounds)
        self.maxiter = maxiter

    def optimize(self, x0, f=None, df=None, f_df=None):
        """
        :param x0: initial point for a local optimizer.
        :param f: function to optimize.
        :param df: gradient of the function to optimize.
        :param f_df: returns both the function to optimize and its gradient.
        """
        try:
            import cma
            def CMA_f_wrapper(f):
                def g(x):
                    return f(np.array([x]))[0][0]
                return g
            lB = np.asarray(self.bounds)[:,0]
            uB = np.asarray(self.bounds)[:,1]
            x = cma.fmin(CMA_f_wrapper(f), x0, 0.6, options={"bounds":[lB, uB], "verbose":-1})[0]
            return np.atleast_2d(x), f(np.atleast_2d(x))
        except ImportError:
            print("Cannot find cma library, please install it to use this option.")
        except:
            print("CMA does not work in problems of dimension 1.")


def apply_optimizer(optimizer, x0, f=None, df=None, f_df=None, duplicate_manager=None, context_manager=None, space=None):
    """
    :param x0: initial point for a local optimizer (x0 can be defined with or without the context included).
    :param f: function to optimize.
    :param df: gradient of the function to optimize.
    :param f_df: returns both the function to optimize and its gradient.
    :param duplicate_manager: logic to check for duplicate (always operates in the full space, context included)
    :param context_manager: If provided, x0 (and the optimizer) operates in the space without the context
    :param space: GPyOpt class design space.
    """
    x0 = np.atleast_2d(x0)

    ## --- Compute a new objective that inputs non context variables but that takes into account the values of the context ones.
    ## --- It does nothing if no context is passed
    problem = OptimizationWithContext(x0=x0, f=f, df=df, f_df=f_df, context_manager=context_manager)


    if context_manager:
        add_context = lambda x : context_manager._expand_vector(x)
    else:
        add_context = lambda x : x

    if duplicate_manager and duplicate_manager.is_unzipped_x_duplicate(x0):
        raise ValueError("The starting point of the optimizer cannot be a duplicate.")

    ## --- Optimize point
    optimized_x, _ = optimizer.optimize(problem.x0_nocontext, problem.f_nocontext, problem.df_nocontext, problem.f_df_nocontext)

    ## --- Add context and round according to the type of variables of the design space
    suggested_x_with_context = add_context(optimized_x)
    suggested_x_with_context_rounded = space.round_optimum(suggested_x_with_context)

    ## --- Run duplicate_manager
    if duplicate_manager and duplicate_manager.is_unzipped_x_duplicate(suggested_x_with_context_rounded):
        suggested_x, suggested_fx = x0, np.atleast_2d(f(x0))
    else:
        suggested_x, suggested_fx = suggested_x_with_context_rounded, f(suggested_x_with_context_rounded)

    return suggested_x, suggested_fx


class OptimizationWithContext(object):

    def __init__(self, x0, f, df=None, f_df=None, context_manager=None):
        '''
        Constructor of an objective function that takes as input a vector x of the non context variables
        and retunrs a value in which the context variables have been fixed.
        '''
        self.x0 = np.atleast_2d(x0)
        self.f = f
        self.df = df
        self.f_df = f_df
        self.context_manager = context_manager

        if not context_manager:
            self.x0_nocontext = x0
            self.f_nocontext  =  self.f
            self.df_nocontext  =  self.df
            self.f_df_nocontext = self.f_df

        else:
            self.x0_nocontext = self.x0[:,self.context_manager.noncontext_index]
            self.f_nocontext  = self.f_nc
            if self.f_df is None:
                self.df_nocontext = None
                self.f_df_nocontext = None
            else:
                self.df_nocontext = self.df_nc
                self.f_df_nocontext  = self.f_df_nc

    def f_nc(self,x):
        '''
        Wrapper of *f*: takes an input x with size of the noncontext dimensions
        expands it and evaluates the entire function.
        '''
        x = np.atleast_2d(x)
        xx = self.context_manager._expand_vector(x)
        if x.shape[0] == 1:
            return self.f(xx)[0]
        else:
            return self.f(xx)

    def df_nc(self,x):
        '''
        Wrapper of the derivative of *f*: takes an input x with size of the not
        fixed dimensions expands it and evaluates the gradient of the entire function.
        '''
        x = np.atleast_2d(x)
        xx = self.context_manager._expand_vector(x)
        _, df_nocontext_xx = self.f_df(xx)
        df_nocontext_xx = df_nocontext_xx[:,np.array(self.context_manager.noncontext_index)]
        return df_nocontext_xx

    def f_df_nc(self,x):
        '''
        Wrapper of the derivative of *f*: takes an input x with size of the not
        fixed dimensions expands it and evaluates the gradient of the entire function.
        '''
        x = np.atleast_2d(x)
        xx = self.context_manager._expand_vector(x)
        f_nocontext_xx , df_nocontext_xx = self.f_df(xx)
        df_nocontext_xx = df_nocontext_xx[:,np.array(self.context_manager.noncontext_index)]
        return f_nocontext_xx, df_nocontext_xx


def choose_optimizer(optimizer_name, bounds):
        """
        Selects the type of local optimizer
        """
        if optimizer_name == 'lbfgs':
            optimizer = OptLbfgs(bounds)

        elif optimizer_name == 'DIRECT':
            optimizer = OptDirect(bounds)

        elif optimizer_name == 'CMA':
            optimizer = OptCma(bounds)
        else:
            raise InvalidVariableNameError('Invalid optimizer selected.')

        return optimizer

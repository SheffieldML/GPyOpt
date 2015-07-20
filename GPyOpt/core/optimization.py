# Copyright (c) 2015, Javier Gonzalez
# Copyright (c) 2015, the GPy Authors (see GPy AUTHORS.txt)
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
import scipy
import GPyOpt
from ..util.general import multigrid, samples_multidimensional_uniform, reshape
from scipy.stats import norm
import numpy as np

# Note this file includes all functions for the optimization of the aquisition functions. This include different batch methods. When data are not
# collected in batches the code goes to predictive_batch_optimization, and runs the first optimization but does not enter in the loop. This 
# should be the same in case of selecting n_inbatch=1 in the rest of the batch methods.


## ---- Predictive batch optimization
def predictive_batch_optimization(acqu_name, acquisition_par, acquisition, d_acquisition, bounds, acqu_optimize_restarts, acqu_optimize_method, model, n_inbatch):   
    '''
    Computes batch optimization using the predictive mean to obtain new batch elements

    :param acquisition: acquisition function in which the batch selection is based
    :param d_acquisition: gradient of the acquisition
    :param bounds: the box constrains of the optimization
    :param acqu_optimize_restarts: the number of restarts in the optimization of the surrogate
    :param acqu_optimize_method: the method to optimize the acquisition function
    :param model: the GP model based on the current samples
    :param n_inbatch: the number of samples to collect
    '''
    model_copy = model.copy()
    X = model_copy.X 
    Y = model_copy.Y
    input_dim = X.shape[1] 
    kernel = model_copy.kern    

    # Optimization of the first element in the batch
    X_new = optimize_acquisition(acquisition, d_acquisition, bounds, acqu_optimize_restarts, acqu_optimize_method, model, X_batch=None, L=None, Min=None)
    X_batch = reshape(X_new,input_dim)
    k=1
    while k<n_inbatch:
        X = np.vstack((X,reshape(X_new,input_dim)))       # update the sample within the batch
        Y = np.vstack((Y,model.predict(reshape(X_new, input_dim))[0]))
       
        try: # this exception is included in case two equal points are selected in a batch, in this case the method stops
            batchBO = GPyOpt.methods.BayesianOptimization(f=0, 
                                        bounds= bounds, 
                                        X=X, 
                                        Y=Y, 
                                        kernel = kernel,
                                        acquisition = acqu_name, 
                                        acquisition_par = acquisition_par)
        except np.linalg.linalg.LinAlgError:
            print 'Optimization stopped. Two equal points selected.'
            break        

        batchBO.run_optimization(max_iter = 0, 
                                    n_inbatch=1, 
                                    acqu_optimize_method = acqu_optimize_method,  
                                    acqu_optimize_restarts = acqu_optimize_restarts, 
                                    eps = 1e-6,verbose = False)
        
        X_new = batchBO.suggested_sample
        X_batch = np.vstack((X_batch,X_new))
        k+=1    
    return X_batch


## ---- Random batch optimization
def random_batch_optimization(acquisition, d_acquisition, bounds, acqu_optimize_restarts, acqu_optimize_method, model, n_inbatch):
    '''
    Computes the batch optimization taking random samples (only for comparative purposes)

    :param acquisition: acquisition function in which the batch selection is based
    :param d_acquisition: gradient of the acquisition
    :param bounds: the box constrains of the optimization
    :param acqu_optimize_restarts: the number of restarts in the optimization of the surrogate
    :param acqu_optimize_method: the method to optimize the acquisition function
    :param model: the GP model based on the current samples
    :param n_inbatch: the number of samples to collect
    '''

    # Optimization of the first element in the batch
    X_batch = optimize_acquisition(acquisition, d_acquisition, bounds, acqu_optimize_restarts, acqu_optimize_method, model)

    k=1 
    while k<n_inbatch:
        new_sample = samples_multidimensional_uniform(bounds,1)
        X_batch = np.vstack((X_batch,new_sample))  
        k +=1
    return X_batch


## ---- Local penalization for batch optimization
def lp_batch_optimization(acquisition, bounds, acqu_optimize_restarts, acqu_optimize_method, model, n_inbatch):
    '''
    Computes batch optimization using by acquisition penalization using Lipschitz inference.

    :param acquisition: acquisition function in which the batch selection is based
    :param d_acquisition: gradient of the acquisition
    :param bounds: the box constrains of the optimization
    :param acqu_optimize_restarts: the number of restarts in the optimization of the surrogate
    :param acqu_optimize_method: the method to optimize the acquisition function
    :param model: the GP model based on the current samples
    :param n_inbatch: the number of samples to collect
    '''
    from .acquisition import AcquisitionMP
    assert isinstance(acquisition, AcquisitionMP)
    acq_func = acquisition.acquisition_function
    d_acq_func = acquisition.d_acquisition_function

    acquisition.update_batches(None,None,None)    
    # Optimize the first element in the batch
    X_batch = optimize_acquisition(acq_func, d_acq_func, bounds, acqu_optimize_restarts, acqu_optimize_method, model, X_batch=None, L=None, Min=None)
    k=1
    #d_acq_func = None  # gradients are approximated  with the batch. 
    
    if n_inbatch>1:
        # ---------- Approximate the constants of the the method
        L = estimate_L(model,bounds)
        Min = estimate_Min(model,bounds)

    while k<n_inbatch:
        acquisition.update_batches(X_batch,L,Min)
        # ---------- Collect the batch (the gradients of the acquisition are approximated for k =2,...,n_inbatch)
        new_sample = optimize_acquisition(acq_func, d_acq_func, bounds, acqu_optimize_restarts, acqu_optimize_method, model, X_batch, L, Min)
        X_batch = np.vstack((X_batch,new_sample))
        k +=1
    acquisition.update_batches(None,None,None)
    return X_batch

def optimize_acquisition(acquisition, d_acquisition, bounds, acqu_optimize_restarts, acqu_optimize_method, model, X_batch=None, L=None, Min=None):
    '''
    Optimization of the acquisition function
    '''
    if acqu_optimize_method=='brute':
        res = full_acquisition_optimization(acquisition, d_acquisition, bounds,acqu_optimize_restarts, model, 'brute', X_batch, L, Min)
    elif acqu_optimize_method=='random':
        res =  full_acquisition_optimization(acquisition, d_acquisition, bounds,acqu_optimize_restarts, model, 'random', X_batch, L, Min)
    elif acqu_optimize_method=='fast_brute':
        res =  fast_acquisition_optimization(acquisition, d_acquisition, bounds,acqu_optimize_restarts, model, 'brute', X_batch, L, Min)
    elif acqu_optimize_method=='fast_random':
        res =  fast_acquisition_optimization(acquisition, d_acquisition, bounds,acqu_optimize_restarts, model, 'random', X_batch, L, Min)
    elif acqu_optimize_method=='DIRECT': 
        res = wrapper_DIRECT(acquisition,bounds)
    elif acqu_optimize_method=='CMA': 
        res = wrapper_CMA(acquisition,bounds)
    return res

def fast_acquisition_optimization(acquisition, d_acquisition, bounds,acqu_optimize_restarts, model, method_type, X_batch=None, L=None, Min=None):
    '''
    Optimizes the acquisition function using a local optimizer in the best point
    '''
    if method_type=='random':
        samples = samples_multidimensional_uniform(bounds,acqu_optimize_restarts)
    else:
        samples = multigrid(bounds, acqu_optimize_restarts)
    pred_samples = acquisition(samples)
    x0 =  samples[np.argmin(pred_samples)]
    best_x,_ = wrapper_lbfgsb(acquisition,d_acquisition,x0 = np.array(x0),bounds=bounds)
    return best_x


def full_acquisition_optimization(acquisition, d_acquisition, bounds, acqu_optimize_restarts, model, method_type, X_batch=None, L=None, Min=None):
    '''
    Optimizes the acquisition function by taking the best of a number of local optimizers
    '''
    if method_type=='random':
        samples = samples_multidimensional_uniform(bounds,acqu_optimize_restarts)
    else:
        samples = multigrid(bounds, acqu_optimize_restarts)
    mins = np.zeros((acqu_optimize_restarts,len(bounds)))
    fmins = np.zeros(acqu_optimize_restarts)
    for k in range(acqu_optimize_restarts):
        mins[k],fmins[k] = wrapper_lbfgsb(acquisition,d_acquisition,x0 = samples[k,:],bounds=bounds)
    return mins[np.argmin(fmins)]


def estimate_L(model,bounds,storehistory=True):
    '''
    Estimate the Lipschitz constant of f by taking maximizing the norm of the expectation of the gradient of *f*.
    '''
    def df(x,model,x0):
        x = reshape(x,model.X.shape[1])
        dmdx,_ = model.predictive_gradients(x)
        res = np.sqrt((dmdx*dmdx).sum(1)) # simply take the norm of the expectation of the gradient
        return -res
   
    samples = samples_multidimensional_uniform(bounds,500)
    samples = np.vstack([samples,model.X])
    pred_samples = df(samples,model,0)
    x0 = samples[np.argmin(pred_samples)]
    res = scipy.optimize.minimize(df,x0, method='L-BFGS-B',bounds=bounds, args = (model,x0), options = {'maxiter': 200})
    minusL = res.fun[0][0]
    L = -minusL
    if L<1e-7: L=10  ## to avoid problems in cases in which the model is flat.
    return L


def estimate_Min(model,bounds):
    '''
    Takes the estimated minimum as the minimum value in the sample. (this function is now nonsense but we will used in further generalizations)
    
    '''
    return model.Y.min()


def wrapper_lbfgsb(f,grad_f,x0,bounds):
    '''
    Wrapper for l-bfgs-b to use the true or the approximate gradients. 
    :param f: function to optimize, acquisition.
    :param grad_f: gradients of f.
    :param x0: initial value for optimization.
    :param bounds: tuple determining the limits of the optimizer.
    '''

    def objective(x):
        return float(f(x)), grad_f(x)[0]

    if grad_f==None:
        res = scipy.optimize.fmin_l_bfgs_b(f, x0=x0, bounds=bounds,approx_grad=True, maxiter=1000)
    else:
        res = scipy.optimize.fmin_l_bfgs_b(objective, x0=x0, bounds=bounds, maxiter=1000)
    return res[0],res[1]


def wrapper_DIRECT(f,bounds):
    '''
    Wrapper for DIRECT optimization method. It works partitioning iteratively the domain 
    of the function. Only requieres f and the box constrains to work
    :param f: function to optimize, acquisition.
    :param bounds: tuple determining the limits of the optimizer.

    '''
    try:
        from DIRECT import solve
        import numpy as np
        def DIRECT_f_wrapper(f):
            def g(x, user_data):
                return f(np.array([x])), 0
            return g
        lB = np.asarray(bounds)[:,0]
        uB = np.asarray(bounds)[:,1]
        x,_,_ = solve(DIRECT_f_wrapper(f),lB,uB, maxT=2000, maxf=2000)
        return reshape(x,len(bounds))
    except:
        print("Cannot find DIRECT library, please install it to use this option.")


def wrapper_CMA(f,bounds):
    '''
    Wrapper the Covariance Matrix Adaptation Evolutionary strategy (CMA-ES) optimization method. It works generating 
    an stochastic seach based on mutivariate Gaussian samples. Only requieres f and the box constrains to work
    :param f: function to optimize, acquisition.
    :param bounds: tuple determining the limits of the optimizer.

    '''
    try:
        import cma 
        import numpy as np
        def CMA_f_wrapper(f):
            def g(x):
                return f(np.array([x]))[0][0]
            return g
        lB = np.asarray(bounds)[:,0]
        uB = np.asarray(bounds)[:,1]
        x = cma.fmin(CMA_f_wrapper(f), (uB + lB) * 0.5, 0.6, options={"bounds":[lB, uB], "verbose":-1})[0]
        print x
        return reshape(x,len(bounds))
    except:
        print("Cannot find cma library, please install it to use this option.")


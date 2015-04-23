import numpy as np
import scipy
import GPyOpt

from ..util.general import multigrid, samples_multidimensional_uniform, reshape
from scipy.stats import norm

##
## ----------- Functions for the optimization of the acquisition function
##


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
                                    stop_criteria = 1e-6,verbose = False)
        
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
def mp_batch_optimization(acquisition, d_acquisition, bounds, acqu_optimize_restarts, acqu_optimize_method, model, n_inbatch):
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

    # Optimize the first element in the batch
    X_batch = optimize_acquisition(acquisition, d_acquisition, bounds, acqu_optimize_restarts, acqu_optimize_method, model, X_batch=None, L=None, Min=None)
    k=1
    d_acquisition = None  # gradients are approximated  with the batch. 
    
    if n_inbatch>1:
        # ---------- Approximate the constants of the the method
        L = estimate_L(model,bounds)
        Min = estimate_Min(model,bounds)    

    while k<n_inbatch:
        # ---------- Collect the batch (the gradients of the acquisition are approximated for k =2,...,n_inbathc)
        new_sample = optimize_acquisition(acquisition, d_acquisition, bounds, acqu_optimize_restarts, acqu_optimize_method, model, X_batch, L, Min)
        X_batch = np.vstack((X_batch,new_sample))  
        k +=1
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
    #elif TODO add DIRECT here
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
    h_func_args = hammer_function_precompute(X_batch, L, Min, model)
    if X_batch==None:
        #res = scipy.optimize.minimize(acquisition, x0=np.array(x0),method='L-BFGS-B',jac=d_acquisition,bounds=bounds, options = {'maxiter': 1000}) 
        best_x,_ = wrapper_lbfgsb(acquisition,d_acquisition,x0 = np.array(x0),bounds=bounds)
    else:
        res = scipy.optimize.minimize(penalized_acquisition, x0=np.array(x0),method='L-BFGS-B',bounds=bounds, args=(acquisition, bounds, model, X_batch)+h_func_args)
        best_x = res.x
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
    h_func_args = hammer_function_precompute(X_batch, L, Min, model)
    for k in range(acqu_optimize_restarts):
        if X_batch==None: # gradients are approximated within the batch collection
            #res = scipy.optimize.minimize(acquisition, x0=samples[k,:],method='L-BFGS-B',jac=d_acquisition,bounds=bounds, options = {'maxiter': 1000})
            mins[k],fmins[k] = wrapper_lbfgsb(acquisition,d_acquisition,x0 = samples[k,:],bounds=bounds)
        else:
            res = scipy.optimize.minimize(penalized_acquisition, x0 = samples[k,:] ,method='L-BFGS-B', bounds=bounds, args=(acquisition, bounds, model, X_batch)+h_func_args)
            mins[k] = res.x
            fmins[k] = res.fun
    return mins[np.argmin(fmins)]


def estimate_L(model,bounds):
    '''
    Estimate the Lipschitz constant of f by taking maximizing the norm of the expectation of the gradient of *f*.
    '''
    def df(x,model,x0):
        x = reshape(x,model.X.shape[1])
        dmdx,_ = model.predictive_gradients(x)
        res = np.sqrt((dmdx*dmdx).sum(1)) # simply take the norm of the expectation of the gradient
        return -res
   
    samples = samples_multidimensional_uniform(bounds,5)
    pred_samples = df(samples,model,0)
    x0 = samples[np.argmin(pred_samples)]
    minusL = scipy.optimize.minimize(df,x0, method='L-BFGS-B',bounds=bounds, args = (model,x0), options = {'maxiter': 1000}).fun[0][0]
    L = -minusL
    if L<0.1: L=100  ## to avoid problems in cases in which the model is flat.
    return L


def estimate_Min(model,bounds):
    '''
    Takes the estimated minimum as the minimum value in the sample
    
    '''
    return model.Y.min()


def hammer_function_precompute(x0, L, Min, model):
    if x0 is None: return None, None
    if len(x0.shape)==1: x0 = x0[None,:]
    m = model.predict(x0)[0]
    pred = model.predict(x0)[1].copy()
    pred[pred<1e-16] = 1e-16
    s = np.sqrt(pred)
    r_x0 = (m-Min)/L
    s_x0 = s/L
    return r_x0, s_x0


def hammer_function(x,x0,r_x0, s_x0):
    '''
    Creates the function to define the exclusion zones
    '''
#    return (norm.cdf((np.sqrt(((x-x0)**2).sum(1))- r_x0)/s_x0)).T
    return norm.cdf((np.sqrt((np.square(np.atleast_2d(x-x0))).sum(1))- r_x0)/s_x0)


def penalized_acquisition(x, acquisition, bounds, model, X_batch, r_x0, s_x0):
    '''
    Creates a penalized acquisition function using 'hammer' functions around the points collected in the batch
    '''
    sur_min = min(-acquisition(model.X))  # assumed minimum of the minus acquisition
    fval = -acquisition(x)-np.sign(sur_min)*(abs(sur_min)) 
    if X_batch!=None:
        h_vals = hammer_function(x, X_batch, r_x0, s_x0)
        fval = fval*np.prod(h_vals)
    return -fval


def wrapper_lbfgsb(f,grad_f,x0,bounds):
    '''
    Wrapper for l-bfgs-b to use the true or the approximate gradients 
    '''

    def objective(x):
        return float(f(x)), grad_f(x)[0]

    if grad_f==None:
        res = scipy.optimize.fmin_l_bfgs_b(f, x0=x0, bounds=bounds,approx_grad=True)
    else:
        res = scipy.optimize.fmin_l_bfgs_b(objective, x0=x0, bounds=bounds)
    
    return res[0],res[1]




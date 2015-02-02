import numpy as np
from ..util.general import multigrid, samples_multidimensional_uniform, reshape, WKmeans
from scipy.stats import norm
import scipy
from scipy import spatial 
import GPyOpt
import random
from functools import reduce
import operator

##
## ----------- General functions for the optimization of the acquisition function
##

def optimize_acquisition(acquisition, bounds, acqu_optimize_restarts, acqu_optimize_method, model, X_batch=None, L=None, Min=None):
    '''
    Optimization of the acquisition function
    '''
    if acqu_optimize_method=='brute':
        res = full_acquisition_optimization(acquisition,bounds,acqu_optimize_restarts, model, 'brute', X_batch, L, Min)
    elif acqu_optimize_method=='random':
        res =  full_acquisition_optimization(acquisition,bounds,acqu_optimize_restarts, model, 'random', X_batch, L, Min)
    elif acqu_optimize_method=='fast_brute':
        res =  fast_acquisition_optimization(acquisition,bounds,acqu_optimize_restarts, model, 'brute', X_batch, L, Min)
    elif acqu_optimize_method=='fast_random':
        res =  fast_acquisition_optimization(acquisition,bounds,acqu_optimize_restarts, model, 'random', X_batch, L, Min)
    return res


def fast_acquisition_optimization(acquisition, bounds,acqu_optimize_restarts, model, method_type, X_batch=None, L=None, Min=None):
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
    res = scipy.optimize.minimize(penalized_acquisition, x0=np.array(x0),method='SLSQP',bounds=bounds, args=(acquisition, bounds, model, X_batch)+h_func_args)
    return res.x


def full_acquisition_optimization(acquisition, bounds, acqu_optimize_restarts, model, method_type, X_batch=None, L=None, Min=None):
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
        res = scipy.optimize.minimize(penalized_acquisition, x0 = samples[k,:] ,method='SLSQP',bounds=bounds, args=(acquisition, bounds, model, X_batch)+h_func_args)
        mins[k] = res.x
        fmins[k] = res.fun
    return mins[np.argmin(fmins)]


##
## ----------- Random batch optimization
##

def random_batch_optimization(acquisition, bounds, acqu_optimize_restarts, acqu_optimize_method, model, n_inbatch):
    '''
    Computes the batch optimization taking random samples (only for comparative purposes)

    '''
    X_batch = optimize_acquisition(acquisition, bounds, acqu_optimize_restarts, acqu_optimize_method, model)
    k=1 
    while k<n_inbatch:
        new_sample = samples_multidimensional_uniform(bounds,1)
        X_batch = np.vstack((X_batch,new_sample))  
        k +=1
    return X_batch


##
## ----------- Maximization-penalization batch optimization
##

def mp_batch_optimization(acquisition, bounds, acqu_optimize_restarts, acqu_optimize_method, model, n_inbatch):
    '''
    Computes batch optimization using by acquisition penalization using Lipschitz inference.

    :param acquisition: acquisition function in which the batch selection is based
    :param bounds: the box constrains of the optimization
    :param restarts: the number of restarts in the optimization of the surrogate
    :param method: the method to optimize the acquisition function
    :param model: the GP model based on the current samples
    :param n_inbatch: the number of samples to collect
    '''
    X_batch = optimize_acquisition(acquisition, bounds, acqu_optimize_restarts, acqu_optimize_method, model, X_batch=None, L=None, Min=None)
    k=1
    if n_inbatch>1:
        # ---------- Approximate the constants of the the method
        L = estimate_L(model,bounds)
        Min = estimate_Min(model,bounds)    

    while k<n_inbatch:
        # ---------- Collect the batch
        new_sample = optimize_acquisition(acquisition, bounds, acqu_optimize_restarts, acqu_optimize_method, model, X_batch, L, Min)
        X_batch = np.vstack((X_batch,new_sample))  
        k +=1
    return X_batch


def estimate_L(model,bounds):
    '''
    Estimate the Lipschitz constant of f by taking maximizing the norm of the expectation of the gradient of *f*.
    '''
    def df(x,model,x0):
        x = reshape(x,model.X.shape[1])
        dmdx, dsdx = model.predictive_gradients(x)
        res = np.sqrt((dmdx*dmdx).sum(1)) # simply take the norm of the expectation of the gradient
        return -res
   
    samples = samples_multidimensional_uniform(bounds,5)
    pred_samples = df(samples,model,0)
    x0 = samples[np.argmin(pred_samples)]
    minusL = scipy.optimize.minimize(df,x0, method='SLSQP',bounds=bounds, args = (model,x0), options = {'maxiter': 1500}).fun[0][0]
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
#        X_batch = reshape(X_batch,model.X.shape[1]) ## TODO: remove this loop
#         for i in range(X_batch.shape[0]):
#             fval = np.multiply(fval,hammer_function(x, X_batch[i,], r_x0, s_x0))
        h_vals = hammer_function(x, X_batch, r_x0, s_x0)
        fval = fval*np.prod(h_vals)
    return -fval


##
## ----------- Predictive batch optimization
##

def predictive_batch_optimization(acqu_name, acquisition_par, acquisition, bounds, acqu_optimize_restarts, acqu_optimize_method, model, n_inbatch):   
    '''
    Computes batch optimization using by acquisition penalization using Lipschitz inference

    :param acquisition: acquisition function in which the batch selection is based
    :param bounds: the box constrains of the optimization
    :param restarts: the number of restarts in the optimization of the surrogate
    :param method: the method to optimize the acquisition function
    :param model: the GP model based on the current samples
    :param n_inbatch: the number of samples to collect
    '''
    model_copy = model.copy()
    X = model_copy.X 
    Y = model_copy.Y
    input_dim = X.shape[1] 
    kernel = model_copy.kern    
    X_new = optimize_acquisition(acquisition, bounds, acqu_optimize_restarts, acqu_optimize_method, model, X_batch=None, L=None, Min=None)
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

        batchBO.start_optimization(max_iter = 0, 
                                    n_inbatch=1, 
                                    acqu_optimize_method = acqu_optimize_method,  
                                    acqu_optimize_restarts = acqu_optimize_restarts, 
                                    stop_criteria = 1e-6,verbose = False)
        
        X_new = batchBO.suggested_sample
        X_batch = np.vstack((X_batch,X_new))
        model_batch = batchBO.model
        k+=1    
    return X_batch


##
## ----------- Simulating and matching batch optimization
##

def sm_batch_optimization(model, n_inbatch, batch_labels):
    n = model.X.shape[0]
    if(n<n_inbatch):
        print 'Initial points should be larger than the batch size'
    weights = np.zeros((n,1))
    X = model.X
    
    ## compute weights
    for k in np.unique(batch_labels):
        x = X[(batch_labels == k)[:,0],:]
        weights[(batch_labels == k)[:,0],:] = compute_batch_weigths(x,model)
        
        ## compute centroids
        X_batch = WKmeans(X,weights,n_inbatch)

        ## perturb points that are equal to already collected locations
        X_batch = perturb_equal_points(model.X,np.vstack(X_batch))
    return X_batch


def compute_w(mu,Sigma):
    n_data = Sigma.shape[0]
    w = np.zeros((n_data,1))
    Sigma12 = scipy.linalg.sqrtm(np.linalg.inv(Sigma)).real
    probabilities = norm.cdf(np.dot(Sigma12,mu))
   
    for i in range(n_data):
        w[i,:] = reduce(operator.mul, np.delete(probabilities,i,0), 1)
    return w

def compute_batch_weigths(x,model):
    Sigma = model.kern.K(x)
    mu = model.predict(x)[0]
    w = compute_w(mu,Sigma)
    return w
    

# test for equally selected points 

def perturb_equal_points(X,X_batch):
    # distance from the points in the batch to the collected points
    min_dist = scipy.spatial.distance.cdist(X,X_batch,'euclidean').min(0)
    input_dim = X.shape[1]

    # indexes of the problematic points
    indexes = [i for i,x in enumerate(min_dist) if x < 1e-9]
            
    # perturb the x
    for k in indexes:
        X_batch [k,:] += np.random.multivariate_normal(np.zeros(input_dim),1e-2*np.eye(input_dim),1).flatten()
    return X_batch




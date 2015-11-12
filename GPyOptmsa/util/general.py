import numpy as np
from scipy.special import erfc

def best_gess(f,X):
    '''
    Gets the best current guess from a vector.
    :param f: function to evaluate.
    :param X: locations.
    '''
    n = X.shape[0]
    xbest = np.zeros(n)
    for i in range(n):
        ff = f(X[0:(i+1)])
        xbest[i] = ff[np.argmin(ff)]
    return xbest


def samples_multidimensional_uniform(bounds,num_data):
    '''
    Generates a multidimensional grid uniformly distributed.
    :param bounds: tuple defining the box constrains.
    :num_data: number of data to generate.

    '''
    dim = len(bounds)
    Z_rand = np.zeros(shape=(num_data,dim))
    for k in range(0,dim): Z_rand[:,k] = np.random.uniform(low=bounds[k][0],high=bounds[k][1],size=num_data)
    return Z_rand


def multigrid(bounds, Ngrid):
    '''
    Generates a multidimensional lattice
    :param bounds: box constrains
    :param Ngrid: number of points per dimension.
    '''
    if len(bounds)==1:
        return np.linspace(bounds[0][0], bounds[0][1], Ngrid).reshape(Ngrid, 1)
    xx = np.meshgrid(*[np.linspace(b[0], b[1], Ngrid) for b in bounds]) 
    return np.vstack([x.flatten(order='F') for x in xx]).T


def reshape(x,input_dim):
    '''
    Reshapes x into a matrix with input_dim columns

    '''
    x = np.array(x)
    if x.size ==input_dim:
        x = x.reshape((1,input_dim))
    return x

def get_moments(model,x):
    '''
    Moments (mean and sdev.) of a GP model at x

    '''
    input_dim = model.input_dim
    x = reshape(x,input_dim)
    fmin = min(model.predict(model.X)[0])
    m, v = model.predict(x)
    s = np.sqrt(np.clip(v, 0, np.inf))
    return (m,s, fmin)

def get_d_moments(model,x):
    '''
    Gradients with respect to x of the moments (mean and sdev.) of the GP
    :param model: GPy model.
    :param x: location where the gradients are evaluated.
    '''
    input_dim = model.input_dim
    x = reshape(x,input_dim)
    _, v = model.predict(x)
    dmdx, dvdx = model.predictive_gradients(x)
    dmdx = dmdx[:,:,0]
    dsdx = dvdx /(2*np.sqrt(v))
    return (dmdx, dsdx)


def get_quantiles(acquisition_par, fmin, m, s):
    '''
    Quantiles of the Gaussian distribution useful to determine the acquisition function values
    :param acquisition_par: parameter of the acquisition function
    :param fmin: current minimum.
    :param m: vector of means.
    :param s: vector of standard deviations. 
    '''
    if isinstance(s, np.ndarray):
        s[s<1e-10] = 1e-10
    elif s< 1e-10:
        s = 1e-10
    u = (fmin-m+acquisition_par)/s
    phi = np.exp(-0.5 * u**2) / np.sqrt(2*np.pi)
    Phi = 0.5 * erfc(-u / np.sqrt(2))
    return (phi, Phi, u)


def best_value(Y,sign=1):
    '''
    Returns a vector whose components i are the minimum (default) or maximum of Y[:i]
    '''
    n = Y.shape[0]
    Y_best = np.ones(n)
    for i in range(n):
        if sign == 1:
            Y_best[i]=Y[:(i+1)].min()
        else:
            Y_best[i]=Y[:(i+1)].max()
    return Y_best

def spawn(f):
    '''
    Function for parallel evaluation of the acquisition function
    '''
    def fun(pipe,x):
        pipe.send(f(x))
        pipe.close()
    return fun








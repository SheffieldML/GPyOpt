# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from scipy.special import erfc
import time
from ..core.errors import InvalidConfigError

def compute_integrated_acquisition(acquisition,x):
    '''
    Used to compute the acquisition function when samples of the hyper-parameters have been generated (used in GP_MCMC model).

    :param acquisition: acquisition function with GpyOpt model type GP_MCMC.
    :param x: location where the acquisition is evaluated.
    '''

    acqu_x = 0

    for i in range(acquisition.model.num_hmc_samples):
        acquisition.model.model.kern[:] = acquisition.model.hmc_samples[i,:]
        acqu_x += acquisition.acquisition_function(x)

    acqu_x = acqu_x/acquisition.model.num_hmc_samples
    return acqu_x

def compute_integrated_acquisition_withGradients(acquisition,x):
    '''
    Used to compute the acquisition function with gradients when samples of the hyper-parameters have been generated (used in GP_MCMC model).

    :param acquisition: acquisition function with GpyOpt model type GP_MCMC.
    :param x: location where the acquisition is evaluated.
    '''

    acqu_x = 0
    d_acqu_x = 0

    for i in range(acquisition.model.num_hmc_samples):
        acquisition.model.model.kern[:] = acquisition.model.hmc_samples[i,:]
        acqu_x_sample, d_acqu_x_sample = acquisition.acquisition_function_withGradients(x)
        acqu_x += acqu_x_sample
        d_acqu_x += d_acqu_x_sample

    acqu_x = acqu_x/acquisition.model.num_hmc_samples
    d_acqu_x = d_acqu_x/acquisition.model.num_hmc_samples

    return acqu_x, d_acqu_x


def best_guess(f,X):
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
    :num_data: number of data points to generate.

    '''
    dim = len(bounds)
    Z_rand = np.zeros(shape=(num_data,dim))
    for k in range(0,dim): Z_rand[:,k] = np.random.uniform(low=bounds[k][0],high=bounds[k][1],size=num_data)
    return Z_rand


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
    input_dim = model.X.shape[1]
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
    dsdx = dvdx / (2*np.sqrt(v))
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
    u = (fmin-m-acquisition_par)/s
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


def evaluate_function(f,X):
    '''
    Returns the evaluation of a function *f* and the time per evaluation
    '''
    num_data, dim_data = X.shape
    Y_eval = np.zeros((num_data, dim_data))
    Y_time = np.zeros((num_data, 1))
    for i in range(num_data):
        time_zero = time.time()
        Y_eval[i,:] = f(X[i,:])
        Y_time[i,:] = time.time() - time_zero
    return Y_eval, Y_time


def values_to_array(input_values):
    '''
    Transforms a values of int, float and tuples to a column vector numpy array
    '''
    if type(input_values)==tuple:
        values = np.array(input_values).reshape(-1,1)
    elif type(input_values) == np.ndarray:
        values = np.atleast_2d(input_values)
    elif type(input_values)==int or type(input_values)==float or type(np.int64):
        values = np.atleast_2d(np.array(input_values))
    else:
        print('Type to transform not recognized')
    return values


def merge_values(values1,values2):
    '''
    Merges two numpy arrays by calculating all possible combinations of rows
    '''
    array1 = values_to_array(values1)
    array2 = values_to_array(values2)

    if array1.size == 0:
        return array2
    if array2.size == 0:
        return array1

    merged_array = []
    for row_array1 in array1:
        for row_array2 in array2:
            merged_row = np.hstack((row_array1,row_array2))
            merged_array.append(merged_row)
    return np.atleast_2d(merged_array)

def round_optimum(x_opt,domain):
    """
    Rounds the some value x_opt to a feasible value in the function domain.
    """

    x_opt_rounded = x_opt.copy()
    counter = 0

    for variable in domain:
        if variable.type == 'continuous':
            var_dim = 1

        elif variable.type == 'discrete':
            var_dim = 1
            x_opt_rounded[0,counter:(counter+var_dim)] = round_discrete(x_opt[0,counter:(counter+var_dim)],variable.domain)

        elif variable.type == 'categorical':
            var_dim = len(variable.domain)
            x_opt_rounded[0,counter:(counter+var_dim)] = round_categorical(x_opt[0,counter:(counter+var_dim)])

        elif variable.type == 'bandit':
            var_dim = variable.domain.shape[1]
            x_opt_rounded[0,counter:(counter+var_dim)] = round_bandit(x_opt[0,counter:(counter+var_dim)],variable.domain)
        else:
            raise Exception('Wrong type of variable')

        counter += var_dim
    return x_opt_rounded


def round_categorical(values):
    """
    Rounds a categorical variable by taking setting to one the max of the given vector and to zero the rest of the entries.
    """

    rounded_values = np.zeros(values.shape)
    rounded_values[np.argmax(values)] = 1
    return rounded_values

def round_discrete(value,domain):
    """
    Rounds a discrete variable by selecting the closest point in the domain
    """
    rounded_value = domain[0]

    for domain_value in domain:
        if np.abs(domain_value-value)< np.abs(rounded_value-value):
            rounded_value = domain_value
    return rounded_value

def round_bandit(value,domain):
    """
    Rounds a discrete variable by selecting the closest point in the domain
    """
    idx = np.argmin(((domain- value)**2).sum(1))
    return domain[idx,:]

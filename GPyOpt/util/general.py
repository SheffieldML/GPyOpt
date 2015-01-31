import numpy as np
from scipy.special import erfc
import random

def best_gess(f,X):
    '''
    Gets the best current guess from a vector and

    '''
    n = X.shape[0]
    xbest = np.zeros(n)
    for i in range(n):
        ff = f(X[0:(i+1)])
        xbest[i] = ff[np.argmin(ff)]
    return xbest


def samples_multidimensional_uniform(bounds,num_data):
    '''
    Generates a multidimensional grid uniformly distributed

    '''
    dim = len(bounds)
    Z_rand = np.zeros(shape=(num_data,dim))
    for k in range(0,dim): Z_rand[:,k] = np.random.uniform(low=bounds[k][0],high=bounds[k][1],size=num_data)
    return Z_rand


def multigrid(bounds, Ngrid):
    '''
    Generates a multidimensional lattice

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
    if len(x.flatten())==input_dim:
        x = x.reshape((1,input_dim))
    return x


def ellipse(points, nstd=2, Nb=100):
    '''
    Generates and ellipse according to the covariance of some data points

    '''
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]
    pos = points.mean(axis=0)
    cov = np.cov(points, rowvar=False)
    vals, vecs = eigsorted(cov)
    theta = np.radians(np.degrees(np.arctan2(*vecs[:,0][::-1])))
    width, height =  nstd * np.sqrt(vals)
    grid = np.linspace(0,2*np.pi,Nb)
    X= width * np.cos(grid)* np.cos(theta) - np.sin(theta) * height * np.sin(grid) + pos[0]
    Y= width * np.cos(grid)* np.sin(theta) + np.cos(theta) * height * np.sin(grid) + pos[1]
    return X,Y


def get_moments(model,x):
    '''
    Moments (mean and sdev.) of a GP model at x

    '''
    input_dim = model.input_dim
    x = reshape(x,input_dim)
    fmin = min(model.predict(model.X)[0])
    m, v = model.predict(x)
    return (m, np.sqrt(np.clip(v, 0, np.inf)), fmin)


def get_quantiles(acquisition_par, fmin, m, s):
    '''
    Quantiles of the Gaussian distribution useful to determine the acquisition function values
    '''
    if isinstance(s, np.ndarray):
        s[s<1e-10] = 1e-10
    elif s< 1e-10:
        s = 1e-10
    u = ((1+acquisition_par)*fmin-m)/s
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

def cluster_points(X, w, mu):
    clusters  = {}
    n = X.shape[0]
    for k in range(n):
        bestmukey = min([(i[0], w[k,:]*np.linalg.norm(X[k,:]-mu[i[0]])) \
                    for i in enumerate(mu)], key=lambda t:t[1])[0]
        try:
            clusters[bestmukey].append(X[k,:])
        except KeyError:
            clusters[bestmukey] = [X[k,:]]
    return clusters
 
def reevaluate_centers(mu, clusters):
    newmu = []
    keys = sorted(clusters.keys())
    for k in keys:
        newmu.append(np.mean(clusters[k], axis = 0))
    return newmu
    
def WKmeans(X,w,n_centroids):
    # Initialize to K random centers
    oldmu = random.sample(X, n_centroids)
    mu = random.sample(X, n_centroids)
    while not has_converged(mu, oldmu):
        oldmu = mu
        # Assign all points in X to clusters
        clusters = cluster_points(X,w, mu)
        # Reevaluate centers
        mu = reevaluate_centers(oldmu, clusters)
    return mu

def has_converged(mu, oldmu):
    return set([tuple(a) for a in mu]) == set([tuple(a) for a in oldmu])










# Copyright (c) 2015, Javier Gonzalez
# Copyright (c) 2015, the GPy Authors (see GPy AUTHORS.txt)
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from ..util.general import reshape

class alpine1:
    '''
    Alpine1 function
    
    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''

    def __init__(self,input_dim, bounds=None, sd=None):
        if bounds == None: 
            self.bounds = bounds  =[(-10,10)]*input_dim
        else: 
            self.bounds = bounds
        self.min = [(0)]*input_dim
        self.fmin = 0
        self.input_dim = input_dim
        if sd==None: 
            self.sd = 0
        else: 
            self.sd=sd

    def f(self,X):
        X = reshape(X,self.input_dim)
        n = X.shape[0]
        fval = (X*np.sin(X) + 0.1*X).sum(axis=1) 
        if self.sd ==0:
            noise = np.zeros(n).reshape(n,1)
        else:
            noise = np.random.normal(0,self.sd,n)
        return fval.reshape(n,1) + noise


class alpine2:
    '''
    Alpine2 function
    
    :param bounds: the box constraints to define the domain in which the function is optimized.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self,input_dim, bounds=None, sd=None):
        if bounds == None: 
            self.bounds = bounds  =[(1,10)]*input_dim
        else: 
            self.bounds = bounds
        self.min = [(7.917)]*input_dim
        self.fmin = 2.808**input_dim
        self.input_dim = input_dim
        if sd==None: 
            self.sd = 0
        else: 
            self.sd=sd

    def f(self,X):
        X = reshape(X,self.input_dim)
        n = X.shape[0]
        fval = np.cumprod(np.sqrt(X),axis=1)[:,self.input_dim-1]*np.cumprod(np.sin(X),axis=1)[:,self.input_dim-1]  
        if self.sd ==0:
            noise = np.zeros(n).reshape(n,1)
        else:
            noise = np.random.normal(0,self.sd,n).reshape(n,1)
        return -fval.reshape(n,1) + noise

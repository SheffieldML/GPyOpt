# Copyright (c) 2016, the GPyOpt Authors
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
        if bounds is  None: 
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
        if bounds is  None: 
            self.bounds = bounds  =[(1,10)]*input_dim
        else: 
            self.bounds = bounds
        self.min = [(7.917)]*input_dim
        self.fmin = -2.808**input_dim
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

class gSobol:
    '''
    gSolbol function
   
    :param a: one-dimensional array containing the coefficients of the function.
    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self,a,bounds=None,sd=None):
        self.a = a
        self.input_dim = len(self.a)

        if bounds is  None: 
            self.bounds =[(-4,6)]*self.input_dim
        else: 
            self.bounds = bounds

        if not (self.a>0).all(): return 'Wrong vector of coefficients, they all should be positive'
        self.S_coef = (1/(3*((1+self.a)**2))) / (np.prod(1+1/(3*((1+self.a)**2)))-1)
        if sd==None: self.sd = 0
        else: self.sd=sd

    def f(self,X):
        X = reshape(X,self.input_dim)
        n = X.shape[0]
        aux = (abs(4*X-2)+np.ones(n).reshape(n,1)*self.a)/(1+np.ones(n).reshape(n,1)*self.a)
        fval =  np.cumprod(aux,axis=1)[:,self.input_dim-1]
        if self.sd ==0:
            noise = np.zeros(n).reshape(n,1)
        else:
            noise = np.random.normal(0,self.sd,n).reshape(n,1)
        return fval.reshape(n,1) + noise

class ackley:
    '''
    Ackley function 

    :param sd: standard deviation, to generate noisy evaluations of the function.
    '''
    def __init__(self, input_dim, bounds=None,sd=None):
        self.input_dim = input_dim

        if bounds is  None: 
            self.bounds =[(-32.768,32.768)]*self.input_dim
        else: 
            self.bounds = bounds

        self.min = [(0.)*self.input_dim]
        self.fmin = 0
        
        if sd==None: 
            self.sd = 0
        else:
            self.sd=sd

    def f(self,X):
        X = reshape(X,self.input_dim)
        print(X)
        n = X.shape[0]
        fval = (20+np.exp(1)-20*np.exp(-0.2*np.sqrt((X**2).sum(1)/self.input_dim))-np.exp(np.cos(2*np.pi*X).sum(1)/self.input_dim))
        
        if self.sd ==0:
            noise = np.zeros(n).reshape(n,1)
        else:
            noise = np.random.normal(0,self.sd,n).reshape(n,1)
        return fval.reshape(n,1) + noise


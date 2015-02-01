from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from ..util.general import reshape

'''
Benchmark of functions of arbitrary dimension for minimization

List of avaiable functions so far:
- Sobol-H .

The classes are oriented to create a python function which contain.
- .f : the funtion itself
- .plot: a plot of the function if the dimension is <=2.
- .min : value of the global minimum(s) for the default parameters.

NOTE: the imput of *f* must be a input_dim numpy array. The dimension is calculated within the fucntion.

Javier Gonzalez August, 2014
'''

class gSobol:
	def __init__(self,a,sd=None):
		self.a = a
		self.input_dim = len(self.a)
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



class alpine1:
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

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

'''
Benchmark of arbitrary dimensional functions interesting to optimize

List of avaiable functions so far:
- Sobol-H .

The classes are oriented to create a python function which contain.
- *.f : the funtion itself
- *.plot: a plot of the function if the dimension is <=2.
- *.sensitivity: The Sobol coefficient per dimension when these are available.
- *.min : value of the global minimum(s) for the default parameters.

NOTE: the imput of .f must be a nxD numpy array. The dimension is calculated within the fucntion.

Javier Gonzalez August, 2014
'''

class gSobol:
	def __init__(self,a,sd=None):
		self.a = a
		self.D = len(self.a)
		if not (self.a>0).all(): return 'Wrong vector of coefficients, they all should be positive'
		self.S_coef = (1/(3*((1+self.a)**2))) / (np.prod(1+1/(3*((1+self.a)**2)))-1)
		if sd==None: self.sd = 0
                else: self.sd=sd
		#self.min = 0.78 
                #self.fmin = -6 

	def f(self,X):
		if len(X.flat)==self.D: X = X.reshape((1,self.D))
		n = X.shape[0]
		aux = (abs(4*X-2)+np.ones(n).reshape(n,1)*self.a)/(1+np.ones(n).reshape(n,1)*self.a)
		fval =  np.cumprod(aux,axis=1)[:,self.D-1]
                if self.sd ==0:
                        noise = np.zeros(n).reshape(n,1)
                else:
                        noise = np.random.normal(0,self.sd,n).reshape(n,1)
                return fval.reshape(n,1) + noise




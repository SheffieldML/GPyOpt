# Copyright (c) 2014, Javier Gonzalez
# Copyright (c) 2014, the GPy Authors (see GPy AUTHORS.txt)
# Licensed under the BSD 3-clause license (see LICENSE.txt)


from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

'''
Benchmark of one dimensional functions interesting to optimize. 

List of available functions so far:
- Forrester

The classes are oriented to create a python function which contain.
- *.f : the function itself
- *.plot: a plot of the function if the dimension is <=2.
- *.min : value of the global minimum(s) for the default parameters.

NOTE: the input of .f must be a nxD numpy array. The dimension is calculated within the function.

Javier Gonzalez August, 2014
'''

class function1d:
	def plot(self,bounds=None):
		if bounds == None: bounds = self.bounds
		X = np.arange(bounds[0][0], bounds[0][1], 0.01)
		Y = self.f(X)
		fig = plt.figure()
		plt.plot(X, Y, lw=2)
                plt.xlabel('x')
                plt.ylabel('f(x)')
		plt.show()

class forrester(function1d):
	def __init__(self,sd=None):
		self.input_dim = 1		
		if sd==None: self.sd = 0
		else: self.sd=sd
		self.min = 0.78 		## approx
		self.fmin = -6 			## approx
		self.bounds = [(0,1)]
                
	def f(self,X):
		X = X.reshape((len(X),1))
		n = X.shape[0]
		fval = ((6*X -2)**2)*np.sin(12*X-4)
		if self.sd ==0:
			noise = np.zeros(n).reshape(n,1)
		else:
			noise = np.random.normal(0,self.sd,n).reshape(n,1)
		return fval.reshape(n,1) + noise



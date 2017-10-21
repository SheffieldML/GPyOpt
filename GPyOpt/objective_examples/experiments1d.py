# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import matplotlib.pyplot as plt
import numpy as np


class function1d:
	'''
	This is a benchmark of unidimensional functions interesting to optimize. 
	'''
	def plot(self, bounds=None, ax=None):
		'''
		:param bounds: the box constraints to define the domain in which
		the function is optimized.
		:param ax: (optional) The axis to plot to
		'''
		if bounds is None:
			bounds = self.bounds
		if ax is None:
			fig = plt.figure()
			ax = fig.add_subplot(1, 1, 1)
			call_show = True
		else:
			call_show = False
		X = np.arange(bounds[0][0], bounds[0][1], 0.01)
		Y = self.f(X)

		ax.plot(X, Y, lw=2, label='f')
		ax.legend()
		ax.set_xlabel('x')
		ax.set_ylabel('f(x)')
		if call_show:
			plt.show()
		return

class forrester(function1d):
	'''Forrester function.
	:param sd: standard deviation, to generate
	  noisy evaluations of the function.
	'''
	def __init__(self,sd=None):
		self.input_dim = 1		
		if sd==None: self.sd = 0
		else: self.sd=sd
		self.min = 0.78			## approx
		self.fmin = -6			## approx
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



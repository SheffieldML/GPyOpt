import numpy as np
import pickle
import cPickle
from ..util.general import  reshape

class gene_optimization:
	def __init__(self,sd=None):
		# load the model
		with open('model_genes_desing') as f:
			model = cPickle.load(f)
		
		# model
		self.model = model 

		# define the bounds
		bounds = [model.X.min(0),model.X.max(1)]
		self.bounds = zip(bounds[0], bounds[1])

		# problem dimension 
		self.input_dim = model.X.shape[1]
		
		# sded of the samples
		if sd==None: self.sd = 0
		else: self.sd=sd

	def f(self,X):
		X = reshape(X,self.input_dim)
		n = X.shape[0]
		fval = self.model.predict(X)[0]
		if self.sd ==0:
			noise = np.zeros(n).reshape(n,1)
		else:
			noise = np.random.normal(0,self.sd,n).reshape(n,1)
		return -fval.reshape(n,1) + noise
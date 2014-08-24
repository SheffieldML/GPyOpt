import GPy
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.special import erfc
from scipy.stats import norm
import matplotlib.pyplot as plt
import scipy
import random

from ..util.general import samples_multimensional_uniform

##
## WORK IN PROGRESS
##


class EntropySearch:
        '''
        Entropy Search optimization. Performs the method proposed in Henning and Schuler (2011)
        
        Javier Gonzalez, August 2014
        - bounds:  list of pairs of tuples defining the box constrains of the optimization
        - kernel:  GPy kernel function. It is set to a RBF if it is not specified
        - T: number of samples in entropy search prediction
	- Ne: number of restarting points for search 
	- Nb: number of representers

        Javier Gonzalez -  August, 2014 
	'''                     

        def __init__(self, bounds=None, kernel=None, optimize_model=None, invertsign=None, Nrandom = None):
	
                if bounds==None:
                        print 'Box contrainst are needed. Please insert box constrains'
                else:
                        self.bounds = bounds
                        self.input_dim = len(self.bounds)
               if kernel is None:
                        self.kernel = GPy.kern.RBF(self.input_dim, variance=.1, lengthscale=.1)
                else:
                        self.kernel = kernel
                if optimize_model == None:
                        self.optimize_model = True
                else:
                        self.optimize_model = optimize_model
                if invertsign == None:
                        self.sign = 1
                else:
                        self.sign = -1
                if Nrandom ==None:
                        self.Nrandom = 5*self.input_dim
                else:
                        self.Nrandom = Nrandom  # 


	def start_optimization(self, f=None, H=None , X=None, Y=None, convergence_plot = True):
		## run optimzation loop for some given number of iterations
		
	
	def continue_optimization(self,H):
		## restart the optimization for more iterations if the method didn't converge



	def sample_belief_locations(self, ...)
		## sample belief locations according to x~ p(f(x)< min f(model))
		##
		## retun zb, mb  (zbelief and the measure mb of those points proportional to p(f(x)< min f(model)) ) 
		

	def join_min(self,...)
		## belief over the minimum of the sampled set


	def current_entropy(self,...)
		## computes the value of the actual entropy
 

	def optimize_entropy(self,...)
		## optimize the entropy and selects the new sample 


	## updates the model (is used after each iteration)
	def update_model(self):
		self.model = GPy.models.GPRegression(self.X,self.Y,self.kernel)
			if self.optimize_model:
				self.model.constrain_positive('')
				self.model.optimize_restarts(num_restarts = 5)
				self.model.optimize()
				self.suggested_sample = self.optimize_acquisition()







import GPy
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from scipy.special import erfc
from scipy.stats import norm
import matplotlib.pyplot as plt
import scipy
import random
from pylab import plot, xlabel, ylabel, title, grid
import matplotlib.pyplot as plt

# this will be replaced by a multidimensional lattice
from ..util.general import samples_multimensional_uniform 


class BayesianOptimization:
	'''
	Standard Bayes Optimzation models in Python.
	
	Javier Gonzalez, August 2014
	- bounds:  list of pairs of tuples defining the box constrains of the optimization
	- kernel:  GPy kernel function. It is set to a RBF if it is not specified
 	- acquisition_type:
		'EI' for expected improvement
       		'MPI' for maximum probability improvement
       		'UCB' for upper confidence band
 	- acquisition_par: parameter of the acquisition function.
 	- grid_search (will be removed by and efficient optimization): Option to do grid search on the acquisition function.
 	- invertsign: By defaulf, minimization is done. Set invertsign to True to maximize
 	- Nrandom: Number of uniform evaluations of f before the optimization starts. If not
	selected if fixed to 5 times the dimensionality of the problem
	
	Javier Gonzalez -  August, 2014 
	'''
	def __init__(self, bounds=None, kernel=None, optimize_model=None, acquisition_type=None, acquisition_par=None, grid_search=None, invertsign=None, Nrandom = None):

		if bounds==None: 
			print 'Box contrainst are needed. Please insert box constrains'	
		else:
			self.bounds = bounds
			self.input_dim = len(self.bounds)
		if acquisition_type == None: 
			self.acquisition_type = 'EI'
		else: 
			self.acquisition_type = acquisition_type 		
		if acquisition_par == None: 
			self.acquisition_par = 0.01
		else: 
			self.acquisition_par = acquisition_par 		
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
	
		if grid_search == None:
			self.grid_search =True
		else:
			self.grid_search = grid.search
		if Nrandom ==None:
			self.Nrandom = 5*self.input_dim
		else: 
			self.Nrandom = Nrandom  # number or samples of random exploration before starting the optimization

		
	def start_optimization(self, f=None, H=None , X=None, Y=None, convergence_plot = True):
		if f==None: print 'Function to optimize is requiered'
		else: self.f = f
		if H == None: H=0
		if X==None or Y == None:
			self.X = samples_multimensional_uniform(self.Nrandom,self.bounds)		 	
			self.Y = f(self.X)
		else:
			self.X = X
			self.Y = Y 
		
 		k=1
		while k<=H:
			self.update_model()
                        self.X = np.vstack((self.X,self.suggested_sample))
                        self.Y = np.vstack((self.Y,self.f(np.array([self.suggested_sample]))))
              		k +=1
			  
		self.update_model()  # last update
		
     #		if convergence_plot == True:
     #			make_convergence_plot()  ## add here plot of the distances d(X[i] - X[i-1])			
     #
		self.optimization_started = True
		return self.suggested_sample
	
	def continue_optimization(self,H):
		if self.optimization_started:
			k=1
			while k<=H:
				self.update_model()
				self.X = np.vstack((self.X,self.suggested_sample))
				self.Y = np.vstack((self.Y,self.f(np.array([self.suggested_sample]))))
				k +=1
			self.update_model()  # last update
			return self.suggested_sample

		else: print 'Optimization not initiated: Use .start_optimization and provide a function to optimize'
		
	def acquisition_function(self,x):
		acquisition_code = {	'EI': self.expected_improvement(x),
					'MPI': self.maximum_probability_improvement(x),
					'UCB': self.upper_confidence_bound(x)}
		return acquisition_code[self.acquisition_type]	
	
	def get_moments(self,x):
		if len(x)==self.input_dim: 
			x = x.reshape((1,self.input_dim))
 		else: 
			x = x.reshape((len(x),self.input_dim)) 
			fmin = min(self.model.predict(self.X)[0])
			m, s = self.model.predict(x)
		return m, s, fmin

	# need to update
	def maximum_probability_improvement(self,x):   
		m, s, fmin = self.get_moments(x) 
		Z = -self.sign * (fmin - m + self.acquisition_par)/s
		f_acqu =  s*Z*norm.cdf(Z) + s * norm.pdf(Z)
		return f_acqu

	# need to update
	def upper_confidence_bound(self,x):	
		m, s, fmin = self.get_moments(x)
		f_acqu = m - self.sign* self.acquisition_par * s
		return f_acqu
	
	# need to update	
 	def expected_improvement(self,x):
		m, s, fmin = self.get_moments(x) 	
		Z = -self.sign * (fmin - m + self.acquisition_par)/s			
		f_acqu = norm.cdf(Z)		
		return f_acqu

# (TO HELP THE OPTIMIZER OF THE ACQUISITION FUNCTION)
#	def d_ maximum_probability_improvement(self,x):
#		m, s, fmin = self.get_moments(x)


#	def d_upper_confidence_bound(self,x):
#		m, s, fmin = self.get_moments(x)


#	def upper_confidence_bound(self,x):
#		m, s, fmin = self.get_moments(x)


	def optimize_acquisition(self):
		if self.grid_search:
			grid = np.linspace(self.bounds[0][0], self.bounds[0][1], 100)
			x = grid[np.argmin(self.acquisition_function(grid))]
		else:
			x0 = np.mean(np.array(self.bounds),axis=1)  # initial value for optimization
			x,f,d = scipy.optimize.fmin_l_bfgs_b(self.acquisition_function, x0=x0, bounds=self.bounds, approx_grad=True )
		return x	
			
	def plot_acquisition(self):
		if self.input_dim ==1:
			X = np.arange(self.bounds[0][0], self.bounds[0][1], 0.001)
          		Y = self.acquisition_function(X)
			Y_normalized = (-Y - min(-Y))/(max(-Y - min(-Y)))
			m, s = self.model.predict(X.reshape(len(X),1))
			## 
			plt.subplot(2, 1, 1)
			plt.plot(self.X, self.Y, 'p')
			plt.plot(X, m, 'b-',lw=2)
			plt.plot(X, m-2*s, 'b-', alpha = 0.5)
			plt.plot(X, m+2*s, 'b-', alpha=0.5)		
			plt.title('Model and observations')
			plt.ylabel('Y values')
			plt.xlabel('X')
			grid(True)
			##
			plt.subplot(2, 1, 2)
			plt.plot(X, Y_normalized, 'r-',lw=2)
			plt.xlabel('time (s)')
			plt.ylabel('Undamped')
			plt.xlabel('X')
			plt.ylabel('Acquisition value')
			plt.title('Acquisition function')
			grid(True)
#		
#		if self.input_dim ==2:

#	def plot_convergence(self):
#
			

	def update_model(self):
		self.model = GPy.models.GPRegression(self.X,self.Y,self.kernel)
                if self.optimize_model:
                        self.model.constrain_positive('')
                        self.model.optimize_restarts(num_restarts = 5)
                        self.model.optimize()
		self.suggested_sample = self.optimize_acquisition()




############






















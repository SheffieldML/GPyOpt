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
from ..util.general import samples_multimensional_uniform, multigrid, reshape, ellipse 


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
	def __init__(self, bounds=None, kernel=None, optimize_model=None, acquisition_type=None, acquisition_par=None, invertsign=None, Nrandom = None):

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
		if Nrandom ==None:
			self.Nrandom = 3*self.input_dim
		else: 
			self.Nrandom = Nrandom  # number or samples of initial random exploration 
		self.Ngrid = 5
		
	def start_optimization(self, f=None, H=None , X=None, Y=None):
		if f==None: print 'Function to optimize is requiered'
		else: self.f = f
		if H == None: H=0
		if X==None or Y == None:
			self.X = samples_multimensional_uniform(self.bounds,self.Nrandom)		 	
			self.Y = f(self.X)
		else:
			self.X = X
			self.Y = Y 
		
		self.update_model()
		prediction = self.model.predict(self.X)
		self.m_in_min = prediction[0]
		self.s_in_min = prediction[1] 
			 
 		k=1
		while k<=H:
			# add new data point in the minumum)
                        self.X = np.vstack((self.X,self.suggested_sample))
                        self.Y = np.vstack((self.Y,self.f(np.array([self.suggested_sample]))))
			pred_min = self.model.predict(reshape(self.suggested_sample,self.input_dim))
			self.m_in_min = np.vstack((self.m_in_min,pred_min[0]))
			self.s_in_min = np.vstack((self.s_in_min,pred_min[1]))
	      		self.update_model()
			k +=1
			  
		self.optimization_started = True
		return self.suggested_sample
	
	def continue_optimization(self,H):
		if self.optimization_started:
			k=1
			while k<=H:
				self.X = np.vstack((self.X,self.suggested_sample))
				self.Y = np.vstack((self.Y,self.f(np.array([self.suggested_sample]))))
				pred_min = self.model.predict(reshape(self.suggested_sample,self.input_dim))
				self.m_in_min = np.vstack((self.m_in_min,pred_min[0]))
				self.s_in_min = np.vstack((self.s_in_min,pred_min[1]))
				self.update_model()				
				k +=1
			return self.suggested_sample

		else: print 'Optimization not initiated: Use .start_optimization and provide a function to optimize'
		
	def acquisition_function(self,x):
		acquisition_code = {	'EI': self.expected_improvement(x),
					'MPI': self.maximum_probability_improvement(x),
					'UCB': self.upper_confidence_bound(x)}
		return acquisition_code[self.acquisition_type]	
	
	def get_moments(self,x):
		x = reshape(x,self.input_dim)
		fmin = min(self.model.predict(self.X)[0])
		m, s = self.model.predict(x)
		return (m, s, fmin)

	def maximum_probability_improvement(self,x):   
		m, s, fmin = self.get_moments(x) 
		u = ((1+self.acquisition_par)*fmin-m)/s
		Phi = 0.5 * erfc(-u / np.sqrt(2))
		f_acqu =  self.sign*Phi
		return -f_acqu # returns negative value for posterior minimization (but we plot f_acqu)

	def upper_confidence_bound(self,x):	
		m, s, fmin = self.get_moments(x)
		f_acqu = self.sign*(-m - self.sign* self.acquisition_par * s)
		return -f_acqu  # returns negative value for posterior minimization (but we plot f_acqu)
	
 	def expected_improvement(self,x):
		m, s, fmin = self.get_moments(x) 	
		u = ((1+self.acquisition_par)*fmin-m)/s	
		phi = np.exp(-0.5 * u**2) / np.sqrt(2*np.pi)
		Phi = 0.5 * erfc(-u / np.sqrt(2))	
		f_acqu = self.sign * (((1+self.acquisition_par)*fmin-m) * Phi + s * phi)
		return -f_acqu  # returns negative value for posterior minimization (but we plot f_acqu)

	def d_maximum_probability_improvement(self,x):
		m, s, fmin = self.get_moments(x)
		u = ((1+self.acquisition_par)*fmin-m)/s	
		phi = np.exp(-0.5 * u**2) / np.sqrt(2*np.pi)
		Phi = 0.5 * erfc(-u / np.sqrt(2))	
		dmdx, dsdx = self.model.predictive_gradients(x)
		df_acqu =  self.sign* ((Phi/s)* (dmdx + dsdx + z))
		return -df_acqu 
		
	def d_upper_confidence_bound(self,x):
		x = reashape(x,self.input_dim)
		dmdx, dsdx = self.model.predictive_gradients(x)		
		df_acqu = self.sign*(-dmdx - self.sign* self.acquisition_par * dsdx) 
		return -df_acqu

	def d_expected_improvement(self,x):
		m, s, fmin = self.get_moments(x)
		u = ((1+self.acquisition_par)*fmin-m)/s	
		phi = np.exp(-0.5 * u**2) / np.sqrt(2*np.pi)
		Phi = 0.5 * erfc(-u / np.sqrt(2))	
		dmdx, dsdx = self.model.predictive_gradients(x)
		df_acqu =  self.sign* (-dmdx * Phi  + dsdx * phi)
		return -df_acqu

	def optimize_acquisition(self):
			# combines initial grid search with local optimzation starting on the minimum of the grid
			grid = multigrid(self.bounds,self.Ngrid)
			pred_grid = self.acquisition_function(grid)
			x0 =  grid[np.argmin(pred_grid)]
			res = scipy.optimize.minimize(self.acquisition_function, x0=np.array(x0), method='SLSQP',bounds=self.bounds)
			return res.x	

	
	def update_model(self):
		self.model = GPy.models.GPRegression(self.X,self.Y,self.kernel)
                if self.optimize_model:
                        self.model.constrain_positive('')
                        self.model.optimize_restarts(num_restarts = 5)
                        self.model.optimize()
		self.suggested_sample = self.optimize_acquisition()

			
	def plot_acquisition(self):
		if self.input_dim ==1:
			X = np.arange(self.bounds[0][0], self.bounds[0][1], 0.001)
          		acqu = self.acquisition_function(X)
			acqu_normalized = (-acqu - min(-acqu))/(max(-acqu - min(-acqu))) # normalize acq in (0,1)
			m, s = self.model.predict(X.reshape(len(X),1))
			plt.figure() 
			plt.subplot(2, 1, 1)
			plt.plot(X, m, 'b-', label=u'Posterior mean',lw=2)
			plt.fill(np.concatenate([X, X[::-1]]), \
        			np.concatenate([m - 1.9600 * s,
                       				(m + 1.9600 * s)[::-1]]), \
        			alpha=.5, fc='b', ec='None', label='95% C. I.')	
			plt.plot(X, m-1.96*s, 'b-', alpha = 0.5)
			plt.plot(X, m+1.96*s, 'b-', alpha=0.5)		
			plt.plot(self.X, self.Y, 'r.', markersize=10, label=u'Observations')
			plt.title('Model and observations')
			plt.ylabel('Y')
			plt.xlabel('X')
			plt.legend(loc='upper left')
			grid(True)
			plt.subplot(2, 1, 2)
			plt.plot(X,acqu_normalized, 'r-',lw=2) 
			plt.xlabel('X')
			plt.ylabel('Acquisition value')
			plt.title('Acquisition function')
			grid(True)

		if self.input_dim ==2:
			X1 = np.linspace(self.bounds[0][0], self.bounds[0][1], 200)
			X2 = np.linspace(self.bounds[1][0], self.bounds[1][1], 200)
			x1, x2 = np.meshgrid(X1, X2)
			X = np.hstack((x1.reshape(200*200,1),x2.reshape(200*200,1)))
			acqu = self.acquisition_function(X)
			acqu_normalized = (-acqu - min(-acqu))/(max(-acqu - min(-acqu)))
			acqu_normalized = acqu_normalized.reshape((200,200))
			m, s = self.model.predict(X)	
			#
			eX3, eY3 = ellipse(self.X,nstd=3)
			eX2, eY2 = ellipse(self.X,nstd=2)
			eX1, eY1 = ellipse(self.X,nstd=1)
			pos = self.X.mean(axis=0)

			##
			plt.figure()
			plt.subplot(1, 3, 1)			
			plt.contourf(X1, X2, m.reshape(200,200),100)
			plt.plot(self.X[:,0], self.X[:,1], 'r.', markersize=10, label=u'Observations')
			plt.colorbar()
			plt.plot(eX1,eY1,"k.-",ms=1,lw=3)
			plt.plot(eX2,eY2,"k.-",ms=1,lw=3)
			plt.plot(eX3,eY3,"k.-",ms=1,lw=3)		
			plt.xlabel('X1')
			plt.ylabel('X2')			
			plt.title('Posterior mean')
			##
			plt.subplot(1, 3, 2)
			plt.plot(self.X[:,0], self.X[:,1], 'r.', markersize=10, label=u'Observations')
			plt.contourf(X1, X2, s.reshape(200,200),100)
			plt.colorbar()
                        plt.plot(eX1,eY1,"k.-",ms=1,lw=3)
                        plt.plot(eX2,eY2,"k.-",ms=1,lw=3)
                        plt.plot(eX3,eY3,"k.-",ms=1,lw=3)
			plt.xlabel('X1')
			plt.ylabel('X2')
			plt.title('Posterior variance')
			##
			plt.subplot(1, 3, 3)
			plt.contourf(X1, X2, acqu_normalized,100)
			plt.colorbar()
                        plt.plot(eX1,eY1,"k.-",ms=1,lw=3)
                        plt.plot(eX2,eY2,"k.-",ms=1,lw=3)
                        plt.plot(eX3,eY3,"k.-",ms=1,lw=3)
			plt.xlabel('X1')
			plt.ylabel('X2')
			plt.title('Acquisition function')



	def plot_convergence(self):
		n = self.X.shape[0]	
		aux = (self.X[1:n,:]-self.X[0:n-1,:])**2		
		distances = np.sqrt(aux.sum(axis=1))

		## plot of distances between consecutive x's
		plt.figure()
		plt.subplot(1, 3, 1)
		plt.plot(range(n-1), distances, '-ro')
		plt.xlabel('Iteration')
		plt.ylabel('d(x[n], x[n-1])')
		plt.title('Distance between consecutive x\'s')
		grid(True)
		# plot of the extimated m(x) at the proposed sampling points
		plt.subplot(1, 3, 2)
		plt.plot(range(self.X.shape[0]),self.m_in_min[:,0],'-o')
                plt.title('GP mean at x[n+1]')
                plt.xlabel('Iteration')
                plt.ylabel('mean at x[n+1]')
                grid(True)
		# Plot of the proposed v(x) at the proposed sampling points
		plt.subplot(1, 3, 3)
		plt.errorbar(range(self.X.shape[0]),[0]*self.X.shape[0] , yerr=self.s_in_min[:,0],fmt='-o',ecolor='b', capthick=1)
		plt.title('GP-model C.I. at x[n+1]')
		plt.xlabel('Iteration')
		plt.ylabel('CI (centered at zero)')
		grid(True)





############






















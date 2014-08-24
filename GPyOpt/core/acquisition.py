import numpy as np

from ..util.general import get_moments, get_quantiles

class AcquisitionBase(object):
	"""
	Base class for acquisition functions in Bayesian Optimization
	"""
	def __init__(self, acquisition_par=None, invertsign=None):
		self.model = None
		if acquisition_par == None: 
			self.acquisition_par = 0.01
		else: 
			self.acquisition_par = acquisition_par 		
		if invertsign == None: 
			self.sign = 1		
		else: 
			self.sign = -1

	def acquisition_function(self, x):
		pass

	def d_acquisition_function(self, x):
		pass


class AcquisitionEI(AcquisitionBase):
	"""
	Class for Expected improvement acquisition functions.
	"""
	def acquisition_function(self,x):
		"""
		Expected Improvement
		"""
		m, s, fmin = get_moments(self.model, x) 	
		phi, Phi, u = get_quantiles(self.acquisition_par, fmin, m, s)	
		f_acqu = self.sign * (((1+self.acquisition_par)*fmin-m) * Phi + s * phi)
		return -f_acqu  # note: returns negative value for posterior minimization 

	def d_acquisition_function(self,x):
		"""
		Derivative of the Expected Improvement
		"""
		m, s, fmin = get_moments(self.model, x)
		phi, Phi, u = get_quantiles(self.acquisition_par, fmin, m, s)	
		dmdx, dsdx = self.model.predictive_gradients(x)
		df_acqu =  self.sign* (-dmdx * Phi  + dsdx * phi)
		return -df_acqu
		

class AcquisitionMPI(AcquisitionBase):
	"""
	Class for Maximum Posterior Improvement acquisition functions.
	"""
	def acquisition_function(self,x):
		"""
		Maximum Posterior Improvement
		"""
		m, s, fmin = get_moments(self.model, x) 	
		phi, Phi, u = get_quantiles(self.acquisition_par, fmin, m, s)	
		f_acqu =  self.sign*Phi
		return -f_acqu  # note: returns negative value for posterior minimization 

	def d_acquisition_function(self,x):
		"""
		Derivative of the Maximum Posterior Improvement
		"""
		m, s, fmin = get_moments(self.model, x)
		phi, Phi, u = get_quantiles(self.acquisition_par, fmin, m, s)	
		dmdx, dsdx = self.model.predictive_gradients(x)
		df_acqu =  self.sign* ((Phi/s)* (dmdx + dsdx + u))
		return -df_acqu


class AcquisitionUCB(AcquisitionBase):
	"""
	Class for Upper Confidence Band acquisition functions.
	"""
	def acquisition_function(self,x):
		"""
		Upper Confidence Band
		"""		
		m, s, fmin = get_moments(self.model, x) 	
		f_acqu = self.sign*(-m - self.sign* self.acquisition_par * s)
		return -f_acqu  # note: returns negative value for posterior minimization 

	def d_acquisition_function(self,x):
		"""
		Derivative of the Upper Confidence Band
		"""
		m, s, fmin = get_moments(self.model, x)
		dmdx, dsdx = self.model.predictive_gradients(x)
		df_acqu = self.sign*(-dmdx - self.sign* self.acquisition_par * dsdx)
		return -df_acqu













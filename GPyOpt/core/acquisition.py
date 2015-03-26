from ..util.general import get_moments, get_quantiles

class AcquisitionBase(object):
	"""
	Base class for acquisition functions in Bayesian Optimization
	"""
	def __init__(self, acquisition_par=None):
		self.model = None
		if acquisition_par == None: 
			self.acquisition_par = 0.01
		else: 
			self.acquisition_par = acquisition_par 		

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
		m, s, fmin, _, _ = get_moments(self.model, x) 	
		phi, Phi, _ = get_quantiles(self.acquisition_par, fmin, m, s)	
		f_acqu = (fmin - m + self.acquisition_par) * Phi + s * phi
		return -f_acqu  # note: returns negative value for posterior minimization 

	def d_acquisition_function(self,x):
		"""
		Derivative of the Expected Improvement (has a very easy derivative!)
		"""
		m, s, fmin, dmdx, dsdx = get_moments(self.model, x) 
		phi, Phi, _ = get_quantiles(self.acquisition_par, fmin, m, s)	
		df_acqu = dsdx * phi - Phi * dmdx
		return -df_acqu
		

class AcquisitionMPI(AcquisitionBase):
	"""
	Class for Maximum Posterior Improvement acquisition functions.
	"""
	def acquisition_function(self,x):
		"""
		Maximum Posterior Improvement
		"""
		m, s, fmin,_,_ = get_moments(self.model, x) 	
		_, Phi,_ = get_quantiles(self.acquisition_par, fmin, m, s)	
		f_acqu =  Phi
		return -f_acqu  # note: returns negative value for posterior minimization 

	def d_acquisition_function(self,x):
		"""
		Derivative of the Maximum Posterior Improvement
		"""
		m, s, fmin, dmdx, dsdx = get_moments(self.model, x) 
		phi, _, u = get_quantiles(self.acquisition_par, fmin, m, s)	
		df_acqu = -(phi/s)* (dmdx + dsdx + u)
		return -df_acqu


class AcquisitionLCB(AcquisitionBase):
	"""
	Class for Upper (lower) Confidence Band acquisition functions.
	"""
	def acquisition_function(self,x):
		"""
		Upper Confidence Band
		"""		
		m, s, _,_,_ = get_moments(self.model, x) 	
		f_acqu = -m + self.acquisition_par * s
		return -f_acqu  # note: returns negative value for posterior minimization 

	def d_acquisition_function(self,x):
		"""
		Derivative of the Upper Confidence Band
		"""
		_, _, _, dmdx, dsdx = get_moments(self.model, x) 
		df_acqu = -dmdx + self.acquisition_par * dsdx
		return -df_acqu











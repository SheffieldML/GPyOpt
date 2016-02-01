import numpy as np
import os
import GPyOpt
import GPy
from GPyOpt.util.general import samples_multidimensional_uniform
from GPyOpt.core.acquisition import AcquisitionEI, AcquisitionEL, AcquisitionMPI, AcquisitionLCB, AcquisitionMP
from utils_test import run_eval
from GPy.models.gradient_checker import GradientChecker
import unittest


class TestAcquisitionsGradients(unittest.TestCase):
	'''
	Unittest for the gradients of the acquisition functions
	'''

	def setUp(self):
		np.random.seed(1)
		self.tolerance 	= 0.01  #Tolerance for difference between true and approximated gradients
		objective 		= GPyOpt.fmodels.experiments1d.forrester()   
		bounds    		= objective.bounds 
		input_dim 		= len(bounds)
		n_inital_design = 8
		X 				= samples_multidimensional_uniform(bounds,n_inital_design)
		Y 				= objective.f(X)
		self.X_test 	= samples_multidimensional_uniform(bounds,n_inital_design)
		
		self.model = GPy.models.GPRegression(X,Y)
		self.model.optimize_restarts(10,verbose=False)
		self.model.Gaussian_noise.constrain_fixed(1e-6, warning=False)

	def test_ChecKGrads_EI(self):
		acquisition_ei = AcquisitionEI()
		acquisition_ei.set_model(self.model)
		grad_ei = GradientChecker(acquisition_ei.acquisition_function,acquisition_ei.d_acquisition_function,self.X_test)
		self.assertTrue(grad_ei.checkgrad(tolerance=self.tolerance))


	def test_ChecKGrads_LCB(self):
		acquisition_lcb = AcquisitionLCB(1)
		acquisition_lcb.set_model(self.model)
		grad_lcb = GradientChecker(acquisition_lcb.acquisition_function,acquisition_lcb.d_acquisition_function,self.X_test)
		self.assertTrue(grad_lcb.checkgrad(tolerance=self.tolerance))


	def test_ChecKGrads_MPI(self):
		acquisition_mpi = AcquisitionMPI(1)
		acquisition_mpi.set_model(self.model)
		grad_mpi = GradientChecker(acquisition_mpi.acquisition_function,acquisition_mpi.d_acquisition_function,self.X_test)
		self.assertTrue(grad_mpi.checkgrad(tolerance=self.tolerance))


if __name__=='main':
	unittest.main()





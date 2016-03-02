from __future__ import print_function

import numpy as np
import os
import unittest

import GPyOpt
from GPyOpt.util.general import samples_multidimensional_uniform
from GPyOpt.testing.utils import run_eval

np.random.seed(1)

class TestAcquisitions(unittest.TestCase):
	'''
	Unittest for the different optimizers of the acquisition functions
	'''

	def setUp(self):

		# -- This file was used to generate the test files
		self.outpath = './test_files'
		self.is_unittest = True	# test files were generated with this line =True

		##
		# -- methods configuration
		##

		# stop conditions
		max_iter 				= 2
		eps 					= 1e-8

		# acquisition type (testing here)
		acquisition_name 		= 'EI'
		acquisition_par			= 0.1

		# acquisition optimization type
		#acqu_optimize_method 	= 'grid'
		acqu_optimize_restarts 	= 5
		true_gradients			= True

		# batch type
		n_inbatch 				= 1
		batch_method			= 'predictive'
		n_procs					= 1

		# type of inital design
		numdata_initial_design 	= 10
		type_initial_design		= 'random'

		# model type
		kernel 					= None
		model_optimize_interval	= 1
		model_optimize_restarts = 2
		sparseGP				= False
		num_inducing			= None

		# likelihood type
		normalize				= False
		exact_feval				= True
		verbosity				= False


		self.methods_configs = [
					{ 'name': 'Grid',
					'max_iter':max_iter,
					'acquisition_name':acquisition_name,
					'acquisition_par': acquisition_par,
					'true_gradients': true_gradients,
					'acqu_optimize_method':'grid',
					'acqu_optimize_restarts':acqu_optimize_restarts,
					'batch_method': batch_method,
					'n_inbatch':n_inbatch,
					'n_procs':n_inbatch,
					'numdata_initial_design': numdata_initial_design,
					'type_initial_design': type_initial_design,
					'kernel': kernel,
					'model_optimize_interval': model_optimize_interval,
					'model_optimize_restarts': model_optimize_restarts,
					'sparseGP': sparseGP,
					'num_inducing': num_inducing,
					'normalize': normalize,
					'exact_feval': exact_feval,
					'eps': eps,
					'verbosity':verbosity,
				  },
				  	{ 'name': 'Brute',
					'max_iter':max_iter,
					'acquisition_name':acquisition_name,
					'acquisition_par': acquisition_par,
					'true_gradients': true_gradients,
					'acqu_optimize_method':'brute',
					'acqu_optimize_restarts':acqu_optimize_restarts,
					'batch_method': batch_method,
					'n_inbatch':n_inbatch,
					'n_procs':n_inbatch,
					'numdata_initial_design': numdata_initial_design,
					'type_initial_design': type_initial_design,
					'kernel': kernel,
					'model_optimize_interval': model_optimize_interval,
					'model_optimize_restarts': model_optimize_restarts,
					'sparseGP': sparseGP,
					'num_inducing': num_inducing,
					'normalize': normalize,
					'exact_feval': exact_feval,
					'eps': eps,
					'verbosity':verbosity,
				  },
				 	{ 'name': 'Random',
					'max_iter':max_iter,
					'acquisition_name':acquisition_name,
					'acquisition_par': acquisition_par,
					'true_gradients': true_gradients,
					'acqu_optimize_method':'random',
					'acqu_optimize_restarts':acqu_optimize_restarts,
					'batch_method': batch_method,
					'n_inbatch':n_inbatch,
					'n_procs':n_inbatch,
					'numdata_initial_design': numdata_initial_design,
					'type_initial_design': type_initial_design,
					'kernel': kernel,
					'model_optimize_interval': model_optimize_interval,
					'model_optimize_restarts': model_optimize_restarts,
					'sparseGP': sparseGP,
					'num_inducing': num_inducing,
					'normalize': normalize,
					'exact_feval': exact_feval,
					'eps': eps,
					'verbosity':verbosity,
				  },
				  	{ 'name': 'Fast_Brute',
					'max_iter':max_iter,
					'acquisition_name':acquisition_name,
					'acquisition_par': acquisition_par,
					'true_gradients': true_gradients,
					'acqu_optimize_method':'fast_brute',
					'acqu_optimize_restarts':acqu_optimize_restarts,
					'batch_method': batch_method,
					'n_inbatch':n_inbatch,
					'n_procs':n_inbatch,
					'numdata_initial_design': numdata_initial_design,
					'type_initial_design': type_initial_design,
					'kernel': kernel,
					'model_optimize_interval': model_optimize_interval,
					'model_optimize_restarts': model_optimize_restarts,
					'sparseGP': sparseGP,
					'num_inducing': num_inducing,
					'normalize': normalize,
					'exact_feval': exact_feval,
					'eps': eps,
					'verbosity':verbosity,
				  },
				  	{'name': 'Fast_Random',
					'max_iter':max_iter,
					'acquisition_name':acquisition_name,
					'acquisition_par': acquisition_par,
					'true_gradients': true_gradients,
					'acqu_optimize_method':'fast_random',
					'acqu_optimize_restarts':acqu_optimize_restarts,
					'batch_method': batch_method,
					'n_inbatch':n_inbatch,
					'n_procs':n_inbatch,
					'numdata_initial_design': numdata_initial_design,
					'type_initial_design': type_initial_design,
					'kernel': kernel,
					'model_optimize_interval': model_optimize_interval,
					'model_optimize_restarts': model_optimize_restarts,
					'sparseGP': sparseGP,
					'num_inducing': num_inducing,
					'normalize': normalize,
					'exact_feval': exact_feval,
					'eps': eps,
					'verbosity':verbosity,
				  },
					{'name': 'DIRECT',
					'max_iter':max_iter,
					'acquisition_name':acquisition_name,
					'acquisition_par': acquisition_par,
					'true_gradients': true_gradients,
					'acqu_optimize_method':'DIRECT',
					'acqu_optimize_restarts':acqu_optimize_restarts,
					'batch_method': batch_method,
					'n_inbatch':n_inbatch,
					'n_procs':n_inbatch,
					'numdata_initial_design': numdata_initial_design,
					'type_initial_design': type_initial_design,
					'kernel': kernel,
					'model_optimize_interval': model_optimize_interval,
					'model_optimize_restarts': model_optimize_restarts,
					'sparseGP': sparseGP,
					'num_inducing': num_inducing,
					'normalize': normalize,
					'exact_feval': exact_feval,
					'eps': eps,
					'verbosity':verbosity,
				  },
				 #  	{'name': 'CMA-ES',
					# 'max_iter':max_iter,
					# 'acquisition_name':acquisition_name,
					# 'acquisition_par': acquisition_par,
					# 'true_gradients': true_gradients,
					# 'acqu_optimize_method':'CMA',
					# 'acqu_optimize_restarts':acqu_optimize_restarts,
					# 'batch_method': batch_method,
					# 'n_inbatch':n_inbatch,
					# 'n_procs':n_inbatch,
					# 'numdata_initial_design': numdata_initial_design,
					# 'type_initial_design': type_initial_design,
					# 'kernel': kernel,
					# 'model_optimize_interval': model_optimize_interval,
					# 'model_optimize_restarts': model_optimize_restarts,
					# 'sparseGP': sparseGP,
					# 'num_inducing': num_inducing,
					# 'normalize': normalize,
					# 'exact_feval': exact_feval,
					# 'eps': eps,
					# 'verbosity':verbosity,
				 #  },
				]

		# -- Problem setup
		np.random.seed(1)
		f_bound_dim = (-5.,5.)
		f_dim = 5
		n_inital_design = 5
		self.f_obj = GPyOpt.fmodels.experimentsNd.gSobol(np.ones(f_dim)).f
		self.f_bounds = [f_bound_dim]*f_dim
		self.f_inits = samples_multidimensional_uniform(self.f_bounds,n_inital_design)
		self.f_inits = self.f_inits.reshape(1, f_dim, self.f_inits.shape[-1])

	def test_run(self):
		for m_c in self.methods_configs:
			np.random.seed(1)
			print('Testing batch method:', m_c['name'])
			name = m_c['name']+'_'+'acquisition_optimizer_testfile'
			unittest_result = run_eval(self.f_obj, self.f_bounds, self.f_inits, method_config=m_c, name=name, outpath=self.outpath, time_limit=None, unittest = self.is_unittest)
			original_result = np.loadtxt(self.outpath +'/'+ name+'.txt')
			self.assertTrue((abs(original_result - unittest_result)<1e-4).all())

if __name__=='main':
	unittest.main()

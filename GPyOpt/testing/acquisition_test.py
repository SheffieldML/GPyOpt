from __future__ import print_function

import numpy as np
import os
import unittest

import GPyOpt
from GPyOpt.util.general import samples_multidimensional_uniform
from GPyOpt.testing.utils import run_eval


class TestAcquisitions(unittest.TestCase):
	'''
	Unittest for the available aquisition functions
	'''

	def setUp(self):

		# -- This file was used to generate the test files
		self.outpath = './test_files'
		self.is_unittest = True	# test files were generated with this line =True

		##
		# -- methods configuration
		##

		# stop conditions
		max_iter 				= 5
		eps 					= 1e-8

		# acquisition type (testing here)
		#acquisition_name 		= 'EI'
		#acquisition_par		= 0

		# acquisition optimization type
		acqu_optimize_method 	= 'grid'
		acqu_optimize_restarts 	= 10
		true_gradients			= True

		# batch type
		n_inbatch 				= 1
		batch_method			= 'predictive'
		n_procs					= 1

		# type of inital design
		numdata_initial_design 	= None
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
					{ 'name': 'EI-0',
					'max_iter':max_iter,
					'acquisition_name':'EI',
					'acquisition_par': 0,
					'true_gradients': true_gradients,
					'acqu_optimize_method':acqu_optimize_method,
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
					{ 'name': 'EI-.1',
					'max_iter':max_iter,
					'acquisition_name':'EI',
					'acquisition_par': .1,
					'true_gradients': true_gradients,
					'acqu_optimize_method':acqu_optimize_method,
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
					{ 'name': 'MPI',
					'max_iter':max_iter,
					'acquisition_name':'MPI',
					'acquisition_par': 0,
					'true_gradients': true_gradients,
					'acqu_optimize_method':acqu_optimize_method,
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
					{ 'name': 'LCB-0.5',
					'max_iter':max_iter,
					'acquisition_name':'LCB',
					'acquisition_par': 0.5,
					'true_gradients': true_gradients,
					'acqu_optimize_method':acqu_optimize_method,
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
		np.random.seed(1)
		for m_c in self.methods_configs:
			print('Testing acquisition ' + m_c['name'])
			name = m_c['name']+'_'+'acquisition_gradient_testfile'
			unittest_result = run_eval(self.f_obj, self.f_bounds, self.f_inits, method_config=m_c, name=name, outpath=self.outpath, time_limit=None, unittest = self.is_unittest)
			original_result = np.loadtxt(self.outpath +'/'+ name+'.txt')
			self.assertTrue((abs(original_result - unittest_result)<1e-4).all())

if __name__=='main':
	unittest.main()

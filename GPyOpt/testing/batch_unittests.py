import numpy as np
import os
import GPyOpt
from GPyOpt.util.general import samples_multidimensional_uniform
from utils_test import run_eval
import unittest

class TestBatchMethods(unittest.TestCase):
	'''
	Unittest for the available batch methods
	'''

	def setUp(self):

		# -- This file was used to generate the test files
		self.outpath = './test_files'
		self.is_unittest = True    # test files were generated with this line =True

		# -- methods configuration
		n_inbatch = 2                         # Number of data per batch
		acquisition_name = 'EI'
		acqu_optimize_method = 'grid'
		acqu_optimize_restarts = 10
		max_iter = 2
		
		self.methods_configs = [
            		{ 'name': 'MP',
            		'acquisition_name':acquisition_name,
                    'acquisition_par': 0,
                    'acqu_optimize_method':acqu_optimize_method,
                    'acqu_optimize_restarts':acqu_optimize_restarts,
                    'batch_method': 'lp',
                    'n_inbatch':n_inbatch,
                    'max_iter':max_iter,
                    'n_procs':n_inbatch,
                    'X-result': 0,
                  },
                  { 'name': 'sequential',
                    'acquisition_name':acquisition_name,
                    'acquisition_par': 0,
                    'acqu_optimize_method':acqu_optimize_method,
                    'acqu_optimize_restarts':acqu_optimize_restarts,
                    'batch_method': 'predictive',
                    'n_inbatch':1,
                    'max_iter':n_inbatch,
                    'n_procs':1,
                    'X-result': 0,
                  },
                 { 'name': 'random',
                   'acquisition_name':acquisition_name,
                    'acquisition_par': 0,
                    'acqu_optimize_method':acqu_optimize_method,
                    'acqu_optimize_restarts':acqu_optimize_restarts,
                    'batch_method': 'random',
                    'n_inbatch':n_inbatch,
                    'max_iter':max_iter,
                    'n_procs':1,
                    'X-result': 0,
                  },
                  { 'name': 'GP-prediction',
                    'acquisition_name':acquisition_name,
                    'acquisition_par': 0,
                    'acqu_optimize_method':acqu_optimize_method,
                    'acqu_optimize_restarts':acqu_optimize_restarts,
                    'batch_method': 'predictive',
                    'n_inbatch':n_inbatch,
                    'max_iter':max_iter,
                    'n_procs':n_inbatch,
                    'X-result': 0,
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
			print 'Testing batch method: ' + m_c['name']
			name = m_c['name']+'_'+'batch_testfile'
			unittest_result = run_eval(self.f_obj, self.f_bounds, self.f_inits, method_config=m_c, name=name, outpath=self.outpath, time_limit=None, unittest = self.is_unittest)            
			original_result = np.loadtxt(self.outpath +'/'+ name+'.txt')
			self.assertTrue((original_result == unittest_result).all())

if __name__=='main':
	unittest.main()




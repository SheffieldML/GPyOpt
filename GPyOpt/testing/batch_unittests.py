import unittest
import numpy as numpy
import GPyOpt
from random numpy inport seed

class TestBatchMethods(unittest.TestCase):
	'''
	Unittest for the available batch methods
	'''

	def SetUp(self):
		n_inbatch = 2                         # Number of data per batch
		n_inital_design = 5
		f_bound_dim = (-5.,5.)
		f_dims = [5]
		acquisition_name = 'EI'
		acqu_optimize_method = 'grid'
		acqu_optimize_restarts = 10
		max_iter = 2

		methods_configs = [
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

    	for f_dim in f_dims:
        	f_obj = GPyOpt.fmodels.experimentsNd.gSobol(np.ones(f_dim)).f
        	f_bounds = [f_bound_dim]*f_dim
        	f_inits = samples_multidimensional_uniform(f_bounds,n_inital_design)
        	f_inits = f_inits.reshape(1, f_dim, f_inits.shape[-1])

        for m_c in methods_configs:
            name = m_c['name']+'_'+'batch_testfile.txt'
            print 'Evaluating '+name+'...'
            m_c['X-result'] = run_eval(f_obj, f_bounds, f_inits, method_config=m_c, name=name, outpath=outpath, time_limit=None,unittest=True)
            #np.loadtxt(outpath+'/'+name+'.txt')

if __name__=='main'
	unittest.main()




# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
import os
import GPyOpt
from GPyOpt.util.general import samples_multidimensional_uniform
from GPyOpt.testing.driver import run_eval
import unittest

class TestAcquisitions(unittest.TestCase):
    '''
    Unittest for the constraints
    '''

    def setUp(self):

        # -- This file was used to generate the test files
        self.outpath = os.path.join(os.path.dirname(__file__),'test_files')
        self.is_unittest = True  # Test files were generated with this line = False

        ##
        # -- methods configuration
        ##

        model_type                  = 'GP' 
        initial_design_numdata      = None
        initial_design_type         = 'random' 
        acquisition_type            = 'EI'
        normalize_Y                 = True 
        exact_feval                 = True
        acquisition_optimizer_type  = 'lbfgs' 
        model_update_interval       = 1 
        evaluator_type              = 'sequential' 
        batch_size                  = 1
        num_cores                   = 1
        verbosity                   = False 

        # stop conditions
        max_iter                    = 15
        max_time                    = 999
        eps                         = 1e-8


        self.methods_configs = [ 
                    {   'name': 'constrains',
                        'model_type'                 : model_type, 
                        'initial_design_numdata'     : initial_design_numdata,
                        'initial_design_type'        : initial_design_type, 
                        'acquisition_type'           : acquisition_type, 
                        'normalize_Y'                : normalize_Y, 
                        'exact_feval'                : exact_feval,
                        'acquisition_optimizer_type' : acquisition_optimizer_type, 
                        'model_update_interval'      : model_update_interval, 
                        'verbosity'                  : verbosity, 
                        'evaluator_type'             : evaluator_type, 
                        'batch_size'                 : batch_size,
                        'num_cores'                  : num_cores,
                        'max_iter'                   : max_iter,
                        'max_time'                   : max_time,
                        'eps'                        : eps
                        }
                    ]

        # -- Problem setup
        np.random.seed(1)
        n_inital_design = 5
        input_dim = 2

        self.problem_config = {
            'objective': GPyOpt.objective_examples.experiments2d.sixhumpcamel().f,
            'domain':      [{'name': 'var_1', 'type': 'continuous', 'domain': (-1,1)},
                            {'name': 'var_2', 'type': 'continuous', 'domain': (-1.5,1.5)}],
            'constrains':  [{'name': 'constr_1', 'constrain': '-x[:,1] -.5 + abs(x[:,0]) - np.sqrt(1-x[:,0]**2)'},
                            {'name': 'constr_2', 'constrain': 'x[:,1] +.5 - abs(x[:,0]) - np.sqrt(1-x[:,0]**2)'}],
            'cost_withGradients': None}


        feasible_region = GPyOpt.Design_space(space = self.problem_config['domain'], constraints = self.problem_config['constrains'])        
        self.f_inits = GPyOpt.util.stats.initial_design('random', feasible_region, 5)
        self.f_inits = self.f_inits.reshape(1,  n_inital_design, input_dim)


    def test_run(self):
        np.random.seed(1)
        for m_c in self.methods_configs:        
            print('Testing acquisition ' + m_c['name'])
            name = m_c['name']+'_'+'acquisition_gradient_testfile'
            unittest_result = run_eval(problem_config= self.problem_config, f_inits= self.f_inits, method_config=m_c, name=name, outpath=self.outpath, time_limit=None, unittest = self.is_unittest)           
            original_result = np.loadtxt(self.outpath +'/'+ name+'.txt')
            self.assertTrue((abs(original_result - unittest_result)<1e-1).all())

if __name__=='main':
    unittest.main()




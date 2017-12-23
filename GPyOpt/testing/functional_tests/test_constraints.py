# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np

import GPyOpt
from GPyOpt.util.general import samples_multidimensional_uniform

from base_test_case import BaseTestCase


class TestAcquisitions(BaseTestCase):
    '''
    Unittest for the constraints
    '''

    def setUp(self):
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
                    {   'name'                       : 'constraints',
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
            'constraints':  [{'name': 'constr_1', 'constraint': '-x[:,1] -.5 + abs(x[:,0]) - np.sqrt(1-x[:,0]**2)'},
                             {'name': 'constr_2', 'constraint': 'x[:,1] +.5 - abs(x[:,0]) - np.sqrt(1-x[:,0]**2)'}],
            'cost_withGradients': None}


        feasible_region = GPyOpt.Design_space(space = self.problem_config['domain'], constraints = self.problem_config['constraints'])
        self.f_inits = GPyOpt.experiment_design.initial_design('random', feasible_region, 5)
        self.f_inits = self.f_inits.reshape(n_inital_design, input_dim)

    def test_run(self):
        self.check_configs()

    def test_run_in_steps(self):
        self.check_configs_in_steps()


if __name__=='main':
    unittest.main()

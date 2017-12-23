# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np

import GPyOpt
from GPyOpt.util.general import samples_multidimensional_uniform

from base_test_case import BaseTestCase


class TestAcquisitions(BaseTestCase):
    '''
    Unittest for the available acquisition functions optimizers
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
        max_iter                    = 5
        max_time                    = 999
        eps                         = 1e-8


        self.methods_configs = [
                    {   'name': 'lbfgs',
                        'model_type'                 : model_type,
                        'initial_design_numdata'     : initial_design_numdata,
                        'initial_design_type'        : initial_design_type,
                        'acquisition_type'           : acquisition_type,
                        'normalize_Y'                : normalize_Y,
                        'exact_feval'                : exact_feval,
                        'acquisition_optimizer_type' : 'lbfgs',
                        'model_update_interval'      : model_update_interval,
                        'verbosity'                  : verbosity,
                        'evaluator_type'             : evaluator_type,
                        'batch_size'                 : batch_size,
                        'num_cores'                  : num_cores,
                        'max_iter'                   : max_iter,
                        'max_time'                   : max_time,
                        'eps'                        : eps
                        },
                    # {   'name': 'DIRECT',
                    #     'model_type'                 : model_type,
                    #     'initial_design_numdata'     : initial_design_numdata,
                    #     'initial_design_type'        : initial_design_type,
                    #     'acquisition_type'           : acquisition_type,
                    #     'normalize_Y'                : normalize_Y,
                    #     'exact_feval'                : exact_feval,
                    #     'acquisition_optimizer_type' : 'DIRECT',
                    #     'model_update_interval'      : model_update_interval,
                    #     'verbosity'                  : verbosity,
                    #     'evaluator_type'             : evaluator_type,
                    #     'batch_size'                 : batch_size,
                    #     'num_cores'                  : num_cores,
                    #     'max_iter'                   : 1,
                    #     'max_time'                   : max_time,
                    #     'eps'                        : eps
                    #     },
                    # {   'name': 'CMA',
                    #     'model_type'                 : model_type,
                    #     'initial_design_numdata'     : initial_design_numdata,
                    #     'initial_design_type'        : initial_design_type,
                    #     'acquisition_type'           : acquisition_type,
                    #     'normalize_Y'                : normalize_Y,
                    #     'exact_feval'                : exact_feval,
                    #     'acquisition_optimizer_type' : 'CMA',
                    #     'model_update_interval'      : model_update_interval,
                    #     'verbosity'                  : verbosity,
                    #     'evaluator_type'             : evaluator_type,
                    #     'batch_size'                 : batch_size,
                    #     'num_cores'                  : num_cores,
                    #     'max_iter'                   : max_iter,
                    #     'max_time'                   : max_time,
                    #     'eps'                        : eps
                    #     }
                    ]

        # -- Problem setup

        n_inital_design = 5
        input_dim = 5
        f_bounds = (-5,5)
        np.random.seed(1)
        self.f_inits = samples_multidimensional_uniform([f_bounds]*input_dim,n_inital_design)
        self.f_inits = self.f_inits.reshape(input_dim, self.f_inits.shape[-1])

        self.problem_config = {
            'objective': GPyOpt.objective_examples.experimentsNd.gSobol(np.ones(input_dim)).f,
            'domain': [{'name': 'var_1', 'type': 'continuous', 'domain': f_bounds, 'dimensionality': input_dim}],
            'constraints': None,
            'cost_withGradients': None}

    def test_run(self):
        self.check_configs()

    def test_run_in_steps(self):
        self.check_configs_in_steps()


if __name__=='main':
    unittest.main()

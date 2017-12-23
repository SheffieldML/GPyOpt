# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np

import GPyOpt
from GPyOpt.util.general import samples_multidimensional_uniform

from base_test_case import BaseTestCase


class TestContextAndParallelization(BaseTestCase):
    '''
    Unittest for the available acquisition functions
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
        batch_size                  = 3
        num_cores                   = 1
        verbosity                   = False

        # stop conditions
        max_iter                    = 1
        max_time                    = 999
        eps                         = 1e-8


        self.methods_configs = [
                    {   'name': 'random_parallel_context_with_duplication',
                        'model_type'                 : model_type,
                        'initial_design_numdata'     : initial_design_numdata,
                        'initial_design_type'        : initial_design_type,
                        'acquisition_type'           : acquisition_type,
                        'normalize_Y'                : normalize_Y,
                        'exact_feval'                : exact_feval,
                        'acquisition_optimizer_type' : acquisition_optimizer_type,
                        'model_update_interval'      : model_update_interval,
                        'verbosity'                  : verbosity,
                        'evaluator_type'             : 'random',
                        'batch_size'                 : batch_size,
                        'num_cores'                  : num_cores,
                        'max_iter'                   : max_iter,
                        'max_time'                   : max_time,
                        'eps'                        : eps,
                        'de_duplication'             : True
                        },
                        {'name': 'random_parallel_context_without_duplication',
                        'model_type'                 : model_type,
                        'initial_design_numdata'     : initial_design_numdata,
                        'initial_design_type'        : initial_design_type,
                        'acquisition_type'           : acquisition_type,
                        'normalize_Y'                : normalize_Y,
                        'exact_feval'                : exact_feval,
                        'acquisition_optimizer_type' : acquisition_optimizer_type,
                        'model_update_interval'      : model_update_interval,
                        'verbosity'                  : verbosity,
                        'evaluator_type'             : 'random',
                        'batch_size'                 : batch_size,
                        'num_cores'                  : num_cores,
                        'max_iter'                   : max_iter,
                        'max_time'                   : max_time,
                        'eps'                        : eps,
                        'de_duplication'             : False
                        },
                        {'name': 'ts_parallel_context_with_duplication',
                        'model_type'                 : model_type,
                        'initial_design_numdata'     : initial_design_numdata,
                        'initial_design_type'        : initial_design_type,
                        'acquisition_type'           : acquisition_type,
                        'normalize_Y'                : normalize_Y,
                        'exact_feval'                : exact_feval,
                        'acquisition_optimizer_type' : acquisition_optimizer_type,
                        'model_update_interval'      : model_update_interval,
                        'verbosity'                  : verbosity,
                        'evaluator_type'             : 'thompson_sampling',
                        'batch_size'                 : batch_size,
                        'num_cores'                  : num_cores,
                        'max_iter'                   : max_iter,
                        'max_time'                   : max_time,
                        'eps'                        : eps,
                        'de_duplication'             : True
                        },
                        {'name': 'ts_parallel_context_without_duplication',
                        'model_type'                 : model_type,
                        'initial_design_numdata'     : initial_design_numdata,
                        'initial_design_type'        : initial_design_type,
                        'acquisition_type'           : acquisition_type,
                        'normalize_Y'                : normalize_Y,
                        'exact_feval'                : exact_feval,
                        'acquisition_optimizer_type' : acquisition_optimizer_type,
                        'model_update_interval'      : model_update_interval,
                        'verbosity'                  : verbosity,
                        'evaluator_type'             : 'thompson_sampling',
                        'batch_size'                 : batch_size,
                        'num_cores'                  : num_cores,
                        'max_iter'                   : max_iter,
                        'max_time'                   : max_time,
                        'eps'                        : eps,
                        'de_duplication'             : False
                        }
                    ]

        # -- Problem setup
        np.random.seed(1)
        n_inital_design = 5
        input_dim = 5

        self.problem_config = {
            'objective': GPyOpt.objective_examples.experimentsNd.alpine1(input_dim = input_dim).f,
            'domain':      [{'name': 'var1', 'type': 'continuous', 'domain': (-10,10),'dimensionality': 2},
                            {'name': 'var3', 'type': 'continuous', 'domain': (-8,3)},
                            {'name': 'var4', 'type': 'categorical', 'domain': (0,1,2)},
                            {'name': 'var5', 'type': 'discrete', 'domain': (-1,5)}],
            'constraints':  None,
            'cost_withGradients': None,
            'context': {'var1_1':0.3,'var4':1}
            }

        feasible_region = GPyOpt.Design_space(space = self.problem_config['domain'], constraints = self.problem_config['constraints'])
        self.f_inits = GPyOpt.experiment_design.initial_design('random', feasible_region, 5)
        self.f_inits = self.f_inits.reshape(n_inital_design, input_dim)

    def test_run(self):
        self.check_configs()


if __name__=='main':
    unittest.main()

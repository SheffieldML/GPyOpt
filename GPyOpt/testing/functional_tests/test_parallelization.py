# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np

import GPyOpt
from GPyOpt.util.general import samples_multidimensional_uniform

from mock import Mock
from mocks import MockModelVectorValuedPredict

from base_test_case import BaseTestCase


class TestAcquisitions(BaseTestCase):
    '''
    Unittest for the available batch methods
    '''

    def setUp(self):

        # Override margin for error
        self.precision = 0.2

        ##
        # -- methods configuration
        ##

        model_type                  = 'GP'
        initial_design_numdata      = None
        initial_design_type         = 'random'
        acquisition_type            = 'EI'
        normalize_Y                 = True
        exact_feval                 = False
        acquisition_optimizer_type  = 'lbfgs'
        model_update_interval       = 1
        batch_size                  = 3
        num_cores                   = 1
        verbosity                   = False

        # stop conditions
        max_iter                    = 2
        max_time                    = 999
        eps                         = 1e-8


        self.methods_configs = [
                    {   'name': 'Random',
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
                        'eps'                        : eps
                        },
                    {   'name': 'Random_with_duplicate_check',
                        'model_type': model_type,
                        'initial_design_numdata': initial_design_numdata,
                        'initial_design_type': initial_design_type,
                        'acquisition_type': acquisition_type,
                        'normalize_Y': normalize_Y,
                        'exact_feval': exact_feval,
                        'acquisition_optimizer_type': acquisition_optimizer_type,
                        'model_update_interval': model_update_interval,
                        'verbosity': verbosity,
                        'evaluator_type': 'random',
                        'batch_size': batch_size,
                        'num_cores': num_cores,
                        'max_iter': max_iter,
                        'max_time': max_time,
                        'eps': eps,
                        'de_duplication' : True
                        },
                    {   'name': 'Local_penalization',
                        'model_type'                 : model_type,
                        'initial_design_numdata'     : initial_design_numdata,
                        'initial_design_type'        : initial_design_type,
                        'acquisition_type'           : acquisition_type,
                        'normalize_Y'                : normalize_Y,
                        'exact_feval'                : exact_feval,
                        'acquisition_optimizer_type' : acquisition_optimizer_type,
                        'model_update_interval'      : model_update_interval,
                        'verbosity'                  : verbosity,
                        'evaluator_type'             : 'local_penalization',
                        'batch_size'                 : batch_size,
                        'num_cores'                  : num_cores,
                        'max_iter'                   : max_iter,
                        'max_time'                   : max_time,
                        'eps'                        : eps
                        },
                    {   'name': 'Thompson_sampling',
                        'model_type': model_type,
                        'initial_design_numdata': initial_design_numdata,
                        'initial_design_type': initial_design_type,
                        'acquisition_type': acquisition_type,
                        'normalize_Y': normalize_Y,
                        'exact_feval': exact_feval,
                        'acquisition_optimizer_type': acquisition_optimizer_type,
                        'model_update_interval': model_update_interval,
                        'verbosity': verbosity,
                        'evaluator_type': 'thompson_sampling',
                        'batch_size': batch_size,
                        'num_cores': num_cores,
                        'max_iter': max_iter,
                        'max_time': max_time,
                        'eps': eps
                        },
                    {   'name': 'Thompson_sampling_with_duplicate_check',
                        'model_type': model_type,
                        'initial_design_numdata': initial_design_numdata,
                        'initial_design_type': initial_design_type,
                        'acquisition_type': acquisition_type,
                        'normalize_Y': normalize_Y,
                        'exact_feval': exact_feval,
                        'acquisition_optimizer_type': acquisition_optimizer_type,
                        'model_update_interval': model_update_interval,
                        'verbosity': verbosity,
                        'evaluator_type': 'thompson_sampling',
                        'batch_size': batch_size,
                        'num_cores': num_cores,
                        'max_iter': max_iter,
                        'max_time': max_time,
                        'eps': eps,
                        'de_duplication': True
                        },
                    ]

        # -- Problem setup
        np.random.seed(1)
        n_inital_design = 5
        input_dim = 5
        f_bounds = (-5,5)
        self.f_inits = samples_multidimensional_uniform([f_bounds]*input_dim,n_inital_design)
        self.f_inits = self.f_inits.reshape(input_dim, self.f_inits.shape[-1])

        self.problem_config = {
            'objective': GPyOpt.objective_examples.experimentsNd.gSobol(np.ones(input_dim)).f,
            'domain': [{'name': 'var_1', 'type': 'continuous', 'domain': f_bounds, 'dimensionality': input_dim}],
            'constraints': None,
            'cost_withGradients': None}

    def test_run(self):
        self.methods_configs = self.get_method_configs_by_evaluators(['local_penalization','thompson_sampling'], include=False)
        self.check_configs()

    def test_thompson_sampling(self):
        self.methods_configs = self.get_method_configs_by_evaluators(['thompson_sampling'], include=True)
        self.check_configs(mock_model = MockModelVectorValuedPredict())

    def test_local_penalization(self):
        gpy_model = Mock()
        gpy_model.X = np.zeros(self.f_inits.shape[1:])
        gpy_model.Y = np.zeros(self.f_inits.shape[1])
        gpy_model.predictive_gradients.side_effect = lambda X: (np.zeros((X.shape[0], X.shape[1], 1)), np.zeros(X.shape))

        self.methods_configs = self.get_method_configs_by_evaluators(['local_penalization'], include=True)
        self.check_configs(mock_gpy_model = gpy_model)

    def test_invalid_mode_type(self):
        self.methods_configs = self.get_method_configs_by_evaluators(['local_penalization'], include=True)
        self.methods_configs[0]['model_type'] = 'RF'

        self.assertRaises(GPyOpt.core.errors.InvalidConfigError, lambda : self.check_configs())

    def test_run_in_steps(self):
        self.methods_configs = self.get_method_configs_by_evaluators(['random'], include=True)
        method_config = self.methods_configs[0]
        original_result = self.load_result_file(method_config['name'])
        num_steps = int((original_result.shape[0] - self.f_inits.shape[1]) / method_config['batch_size'])

        self.methods_configs = [method_config]
        self.check_configs_in_steps(init_num_steps = num_steps)

    def test_local_penalization_in_steps(self):
        gpy_model = Mock()
        gpy_model.X = np.zeros(self.f_inits.shape[1:])
        gpy_model.Y = np.zeros(self.f_inits.shape[1])
        gpy_model.predictive_gradients.side_effect = lambda X: (np.zeros((X.shape[0], X.shape[1], 1)), np.zeros(X.shape))

        self.methods_configs = self.get_method_configs_by_evaluators(['local_penalization'], include=True)
        method_config = self.methods_configs[0]
        original_result = self.load_result_file(method_config['name'])
        num_steps = int((original_result.shape[0] - self.f_inits.shape[1]) / method_config['batch_size'])

        self.check_configs_in_steps(mock_gpy_model = gpy_model, init_num_steps = num_steps)

    def get_method_configs_by_evaluators(self, evaluators, include=True):
        '''
        Return the sublist of method configs filtered by evaluator types
        :param evaluators: list of evaluators
        :param include: return entries that match the list provided if true, entries that do not match otherwise
        '''
        return [mc for mc in self.methods_configs if (include and mc['evaluator_type'] in evaluators) or (not include and mc['evaluator_type'] not in evaluators)]

if __name__=='main':
    unittest.main()

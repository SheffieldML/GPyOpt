import os
import numpy as np

import unittest
from mock import patch

from driver import run_eval, run_evaluation_in_steps
from mocks import MockModel

class BaseTestCase(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(BaseTestCase, self).__init__(*args, **kwargs)

        # This file was used to generate the test files
        self.outpath = os.path.join(os.path.dirname(__file__), 'test_files')

        # Change this False to generate test files
        self.is_unittest = True

        # Allowed margin of error for test outputs
        self.precision = 1e-6

    def get_result_filename(self, test_name):
        return '{}_{}'.format(test_name, 'acquisition_gradient_testfile')

    def load_result_file(self, test_name):
        filename = self.get_result_filename(test_name)
        file_path = '{}/{}.txt'.format(self.outpath, filename)
        original_result = np.loadtxt(file_path)
        return original_result

    @patch('GPyOpt.methods.BayesianOptimization._model_chooser')
    def check_configs(self, mock_model_chooser, mock_gpy_model = None, mock_model = MockModel()):
        if mock_gpy_model is not None:
            mock_model.model = mock_gpy_model
        mock_model_chooser.return_value = mock_model

        for m_c in self.methods_configs:
            np.random.seed(1)

            if mock_gpy_model is not None:
                mock_model.model = mock_gpy_model
            mock_model_chooser.return_value = mock_model

            print('Testing acquisition ' + m_c['name'])
            name = self.get_result_filename(m_c['name'])
            unittest_result = run_eval(problem_config= self.problem_config, f_inits= self.f_inits, method_config=m_c, name=name, outpath=self.outpath, time_limit=None, unittest = self.is_unittest)
            original_result = self.load_result_file(m_c['name'])

            self.assertTrue((abs(original_result - unittest_result) < self.precision).all(), msg=m_c['name'] + ' failed')

    @patch('GPyOpt.methods.BayesianOptimization._model_chooser')
    def check_configs_in_steps(self, mock_model_chooser, mock_gpy_model=None, init_num_steps=None):
        for m_c in self.methods_configs:
            np.random.seed(1)
            mock_model = MockModel()
            if mock_gpy_model is not None:
                mock_model.model = mock_gpy_model
            mock_model_chooser.return_value = mock_model

            print('Testing acquisition ' + m_c['name'] + ' in steps')
            original_result = self.load_result_file(m_c['name'])

            if init_num_steps is None:
                num_steps = original_result.shape[0] - self.f_inits.shape[0]
            else:
                num_steps = init_num_steps

            unittest_result = run_evaluation_in_steps(problem_config= self.problem_config, f_inits= self.f_inits, method_config=m_c, num_steps=num_steps)

            self.assertTrue((abs(original_result - unittest_result) < self.precision).all(), msg=m_c['name'] + ' failed step-by-step check')

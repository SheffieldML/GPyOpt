import os
import numpy as np

import unittest
from mock import patch

from driver import run_eval
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

    @patch('GPyOpt.methods.BayesianOptimization._model_chooser')
    def check_configs(self, mock_model_chooser, mock_gpy_model = None, mock_model = MockModel()):
        if mock_gpy_model is not None:
            mock_model.model = mock_gpy_model
        mock_model_chooser.return_value = mock_model


        for m_c in self.methods_configs:
            np.random.seed(1)
            print('Testing acquisition ' + m_c['name'])
            name = '{}_{}'.format(m_c['name'], 'acquisition_gradient_testfile')

            unittest_result = run_eval(problem_config= self.problem_config, f_inits= self.f_inits, method_config=m_c, name=name, outpath=self.outpath, time_limit=None, unittest = self.is_unittest)
            original_result = np.loadtxt(self.outpath +'/'+ name+'.txt')

            self.assertTrue((abs(original_result - unittest_result) < self.precision).all(), msg=m_c['name'] + ' failed')

import unittest
from mock import Mock

import numpy as np

from GPyOpt.acquisitions import AcquisitionLCB
from GPyOpt.core.task.space import Design_space

class TestLCBAcquisition(unittest.TestCase):
    def setUp(self):
        self.mock_model = Mock()
        self.mock_optimizer = Mock()
        domain = [{'name': 'var_1', 'type': 'continuous', 'domain': (-5,5), 'dimensionality': 2}]
        self.space = Design_space(domain, None)

        self.lcb_acquisition = AcquisitionLCB(self.mock_model, self.space, self.mock_optimizer)

    def test_acquisition_function(self):
        self.mock_model.predict.return_value = (1, 3)

        weighted_acquisition = self.lcb_acquisition.acquisition_function(np.array([2,2]))
        expected_acquisition = np.array([[-5.0],[-5.0]])

        self.assertTrue(np.array_equal(expected_acquisition, weighted_acquisition))

    def test_acquisition_function_withGradients(self):
        self.mock_model.predict_withGradients.return_value = (1, 1, 0.1, 0.1)

        weighted_acquisition, weighted_gradient = self.lcb_acquisition.acquisition_function_withGradients(np.array([2,2]))

        self.assertTrue(np.array_equal(np.array([[-1.0],[-1.0]]), weighted_acquisition))
        self.assertTrue(np.array_equal(np.array([[-0.1,-0.1],[-0.1,-0.1]]), weighted_gradient))

    def test_optimize_with_analytical_gradient_prediction(self):
        expected_optimum_position = [[0,0]]
        self.mock_optimizer.optimize.return_value = expected_optimum_position
        self.mock_model.analytical_gradient_prediction = True

        self.lcb_acquisition = AcquisitionLCB(self.mock_model, self.space, self.mock_optimizer)
        optimum_position = self.lcb_acquisition.optimize()

        self.assertEqual(expected_optimum_position, optimum_position)

    def test_optimize_without_analytical_gradient_prediction(self):
        expected_optimum_position = [[0,0]]
        self.mock_optimizer.optimize.return_value = expected_optimum_position
        self.mock_model.analytical_gradient_prediction = False
        self.lcb_acquisition = AcquisitionLCB(self.mock_model, self.space, self.mock_optimizer)
        
        optimum_position = self.lcb_acquisition.optimize()

        self.assertEqual(expected_optimum_position, optimum_position)

import unittest
from mock import Mock

import numpy as np

from GPyOpt.acquisitions import AcquisitionMPI
from GPyOpt.core.task.space import Design_space

class TestMPIAcquisition(unittest.TestCase):
    def setUp(self):
        self.mock_model = Mock()
        self.mock_optimizer = Mock()
        domain = [{'name': 'var_1', 'type': 'continuous', 'domain': (-5,5), 'dimensionality': 2}]
        self.space = Design_space(domain, None)

        self.mpi_acquisition = AcquisitionMPI(self.mock_model, self.space, self.mock_optimizer)
        
    def test_acquisition_function(self):
        self.mock_model.predict.return_value = (1,3)
        self.mock_model.get_fmin.return_value = (0.1)

        weighted_acquisition = self.mpi_acquisition.acquisition_function(np.array([2,2]))
        
        expected_acquisition = np.array([[-0.38081792], [-0.38081792]])
        self.assertTrue(np.isclose(weighted_acquisition, expected_acquisition).all())
        
    def test_acquisition_function_withGradients(self):
        self.mock_model.predict_withGradients.return_value = (1,3,0.1,0.2)
        self.mock_model.get_fmin.return_value = 0.1
        
        weighted_acquisition, weighted_gradient = self.mpi_acquisition.acquisition_function_withGradients(np.array([2,2]))

        self.assertTrue(np.isclose(weighted_acquisition, np.array([[-0.38081792],[-0.38081792]])).all())
        self.assertTrue(np.isclose(weighted_gradient, np.array([[0.00499539, 0.00499539],[0.00499539,0.00499539]])).all())
        
    def test_optimize_with_analytical_gradient_prediction(self):
        expected_optimum_position = [[0,0]]
        self.mock_optimizer.optimize.return_value = expected_optimum_position
        self.mock_model.analytical_gradient_prediction = True
        self.mpi_acquisition = AcquisitionMPI(self.mock_model, self.space, self.mock_optimizer)

        optimum_position = self.mpi_acquisition.optimize()

        self.assertEqual(optimum_position, expected_optimum_position)

    def test_optimize_without_analytical_gradient_prediction(self):
        expected_optimum_position = [[0,0]]
        self.mock_optimizer.optimize.return_value = expected_optimum_position
        self.mock_model.analytical_gradient_prediction = False
        self.mpi_acquisition = AcquisitionMPI(self.mock_model, self.space, self.mock_optimizer)

        optimum_position = self.mpi_acquisition.optimize()

        self.assertEqual(optimum_position, expected_optimum_position)

import unittest
from mock import Mock

import numpy as np

from GPyOpt.acquisitions import AcquisitionEI
from GPyOpt.core.task.space import Design_space

class TestEIAcquisition(unittest.TestCase):
    def setUp(self):
        self.mock_model = Mock()
        self.mock_optimizer = Mock()
        domain = [{'name': 'var_1', 'type': 'continuous', 'domain': (-5,5), 'dimensionality': 2}]
        self.space = Design_space(domain, None)

        self.ei_acquisition = AcquisitionEI(self.mock_model, self.space, self.mock_optimizer)

    def test_acquisition_function(self):
        """Test that acquisition function returns correct weighted acquisition
        """
        self.mock_model.predict.return_value = (1, 3)
        self.mock_model.get_fmin.return_value = 0.1

        weighted_acquisition = self.ei_acquisition.acquisition_function(np.array([2,2]))

        assert np.isclose(weighted_acquisition, np.array([[-0.79646919], [-0.79646919]])).all()

    def test_acquisition_function_withGradients(self):
        """Test that acquisition function with gradients returns correct weight acquisition and gradient
        """
        self.mock_model.predict_withGradients.return_value = (1, 1, 0.1, 0.1)
        self.mock_model.get_fmin.return_value = 0.1

        weighted_acquisition, weighted_gradient = self.ei_acquisition.acquisition_function_withGradients(np.array([2,2]))

        assert np.isclose(weighted_acquisition, np.array([[-0.0986038],[-0.0986038]])).all()
        assert np.isclose(weighted_gradient, np.array([[-0.00822768, -0.00822768], [-0.00822768, -0.00822768]])).all()

    def test_optimize_with_analytical_gradient_prediction(self):
        """Test that acquisition function optimize method returns expected optimum with analytical gradient prediction
        """
        expected_optimum_position = [[0, 0]]
        self.mock_optimizer.optimize.return_value = expected_optimum_position
        self.mock_model.analytical_gradient_prediction = True
        self.ei_acquisition = AcquisitionEI(self.mock_model, self.space, self.mock_optimizer)

        optimum_position = self.ei_acquisition.optimize()

        assert optimum_position == expected_optimum_position

    def test_optimize_without_analytical_gradient_prediction(self):
        """Test that acquisition function optimize method returns expected optimum without analytical gradient prediction
        """
        expected_optimum_position = [[0, 0]]
        self.mock_optimizer.optimize.return_value = expected_optimum_position
        self.mock_model.analytical_gradient_prediction = False
        self.ei_acquisition = AcquisitionEI(self.mock_model, self.space, self.mock_optimizer)

        optimum_position = self.ei_acquisition.optimize()

        assert optimum_position == expected_optimum_position
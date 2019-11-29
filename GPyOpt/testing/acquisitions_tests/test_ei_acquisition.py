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


class TestEIAcquisitionWithCategoricalVariables(unittest.TestCase):
	def setUp(self):
		self.mock_model = Mock()
		self.mock_optimizer = Mock()
		
		domain = [{'name': 'var_1', 'type': 'categorical', 'domain': (0, 1)}, {'name': 'var_2', 'type': 'continuous', 'domain': (-5,5), 'dimensionality': 2}]
		# con_1: if var_1 is 0, var_2_1 must be <= -1
		# con_2: 3 * (var_2_1 + var_2_2) <= 24
		constraints = [{'name': 'con_1', 'constraint': '(x[:,0] == 0) * (x[:,1] + 1)'}, 
						{'name': 'con_2', 'constraint': ' 3 * (x[:,1] + x[:,2]) - 24'}]
		self.space = Design_space(domain, constraints)

		self.ei_acquisition = AcquisitionEI(self.mock_model, self.space, self.mock_optimizer)
		self.ei_acquisition._compute_acq = Mock()
		self.ei_acquisition._compute_acq_withGradients = Mock()

	def test_acquisition_function(self):
		"""Test that acquisition function does correct constraint(s) check"""
		
		y = [1.37, 8.22, 4.2, 0.55, 3.14]
		self.ei_acquisition._compute_acq.return_value = np.array(y)[:, None]
		correct_y = [-y[0]] + [0,0] + [-y[3]] + [0]
		x_unzipped = np.array([[0, 1, 3.3, -3.3], [1, 0, 1.5, 4.7], [0, 1, 4.1, 4.5], [1, 0, -4.1, 4.5], [1, 0, 5, 3]])
		
		acquisitions = self.ei_acquisition.acquisition_function(x_unzipped)
		
		assert np.isclose(acquisitions, np.array(correct_y)[:, None]).all()
	
	def test_acquisition_function_withGradients(self):
		"""Test that acquisition function does correct constraint(s) check with gradients"""

		y = [1.37, 8.22, 4.2, 0.55, 3.14]
		y_grad = [.3, .7, -.5, .1, -.02]
		self.ei_acquisition._compute_acq_withGradients.return_value = np.array(y)[:, None], np.array(y_grad)[:, None]
		correct_y = [-y[0]] + [0,0] + [-y[3]] + [0]
		correct_y_grad = [-y_grad[0]] + [0,0] + [-y_grad[3]] + [0]
		x_unzipped = np.array([[0, 1, 3.3, -3.3], [1, 0, 1.5, 4.7], [0, 1, 4.1, 4.5], [1, 0, -4.1, 4.5], [1, 0, 5, 3]])

		acquisitions, gradients = self.ei_acquisition.acquisition_function_withGradients(x_unzipped)

		assert np.isclose(acquisitions, np.array(correct_y)[:, None]).all()
		assert np.isclose(gradients, np.array(correct_y_grad)[:, None]).all()

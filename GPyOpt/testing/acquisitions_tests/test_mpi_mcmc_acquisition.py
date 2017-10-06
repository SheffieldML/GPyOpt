import unittest
import numpy as np

from GPyOpt.acquisitions import AcquisitionMPI_MCMC
from GPyOpt.core.task.space import Design_space
from mock import Mock

class TestMPImcmcAcquisition(unittest.TestCase):
    def setUp(self):
        self.mock_model = Mock()
        self.mock_optimizer = Mock()
        domain = [{'name': 'var_1', 'type': 'continuous', 'domain': (-5,5), 'dimensionality': 2}]
        self.space = Design_space(domain, None)

        self.mpi_mcmc_acquisition = AcquisitionMPI_MCMC(self.mock_model, self.space, self.mock_optimizer)
        
    def test_acquisition_function(self):
        """Test that acquisition function returns correct weighted acquisition
        """
        self.mock_model.predict.return_value = np.array([[1,2,3,4], [3,3,3,3]])
        self.mock_model.get_fmin.return_value = np.array([[0.1,0.2,0.3,0.4]])
        
        weighted_acquisition = self.mpi_mcmc_acquisition.acquisition_function(np.array([2,2]))
        
        expected_acquisition = np.array([[-0.09520448, -0.09839503, -0.10161442, -0.10485931], [-0.09520448, -0.09839503, -0.10161442, -0.10485931]])
        self.assertTrue(np.isclose(weighted_acquisition, expected_acquisition).all())

    def test_acquisition_function_withGradients(self):
        """Test that acquisition function with gradients returns correct weight acquisition and gradient
        """
        self.mock_model.predict_withGradients.return_value = np.array([[1,2,3,4],[3,2,3,5],[0.1,0.4,0.1,0.2],[0.2,0.4,0.2,0.1]])
        self.mock_model.get_fmin.return_value = np.array([[1,3]])
        
        weighted_acquisition, weighted_gradient = self.mpi_mcmc_acquisition.acquisition_function_withGradients(np.array([2,2]))

        self.assertTrue(np.isclose(weighted_acquisition, np.array([[-0.12466755, -0.18661036],[-0.12466755, -0.18661036]])).all())
        self.assertTrue(np.isclose(weighted_gradient, np.array([[0.00330234, 0.00620749],[0.00330234, 0.00620749]])).all())

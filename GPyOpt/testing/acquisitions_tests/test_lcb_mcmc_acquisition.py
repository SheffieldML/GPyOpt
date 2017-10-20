import unittest
from mock import Mock

import numpy as np

from GPyOpt.acquisitions import AcquisitionLCB_MCMC
from GPyOpt.core.task.space import Design_space

class TestLCBmcmcAcquisition(unittest.TestCase):
    def setUp(self):
        self.mock_model = Mock()
        self.mock_optimizer = Mock()
        domain = [{'name': 'var_1', 'type': 'continuous', 'domain': (-5,5), 'dimensionality': 2}]
        self.space = Design_space(domain, None)
        
        self.lcb_mcmc_acquisition = AcquisitionLCB_MCMC(self.mock_model, self.space, self.mock_optimizer)
        
    def test_acquisition_function(self):
        self.mock_model.predict.return_value = ([1,2,3,4],[1,.5,0,2])
        
        weighted_acquisition = self.lcb_mcmc_acquisition.acquisition_function(np.array([2,2]))
        
        self.assertTrue(np.array_equal(np.array([[0.75],[0.75]]), weighted_acquisition))

    def test_acquisition_function_withGradients(self):
        self.mock_model.predict_withGradients.return_value = ([1.,2.,3.,4.],[3.,2.,3.,2.],[0.1,0.1,0.1,0.1],[0.2,0.2,0.2,0.2])
        
        weighted_acquisition, weighted_gradient = self.lcb_mcmc_acquisition.acquisition_function_withGradients(np.array([2.,2.]))
    
        expected_acquisition = np.array([[-2.5],[-2.5]])
        expected_gradient = np.array([[-0.3,-0.3],[-0.3,-0.3]])
    
        self.assertTrue(np.array_equal(expected_acquisition, weighted_acquisition))
        self.assertTrue(np.isclose(weighted_gradient,expected_gradient).all())

    def test_optimize_with_analytical_gradient_prediction(self):
        expected_optimum_position = [[0,0]]
        self.mock_optimizer.optimize.return_value = expected_optimum_position
        self.mock_model.analytical_gradient_prediction = True
        self.lcb_mcmc_acquisition = AcquisitionLCB_MCMC(self.mock_model, self.space, self.mock_optimizer)
        
        optimum_position = self.lcb_mcmc_acquisition.optimize()
        
        self.assertEqual(expected_optimum_position, optimum_position)
        
    def test_optimize_without_analytical_gradient_prediction(self):
        expected_optimum_position = [[0,0]]
        self.mock_optimizer.optimize.return_value = expected_optimum_position
        self.mock_model.analytical_gradient_prediction = False
        self.lcb_mcmc_acquisition = AcquisitionLCB_MCMC(self.mock_model, self.space, self.mock_optimizer)
        
        optimum_position = self.lcb_mcmc_acquisition.optimize()
        
        self.assertEqual(expected_optimum_position, optimum_position)
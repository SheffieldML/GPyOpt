import unittest
from mock import Mock

import numpy as np

from GPyOpt.core.evaluators import RandomBatch
from GPyOpt.acquisitions import AcquisitionEI
from GPyOpt.core.task.space import Design_space
from GPyOpt.optimization.acquisition_optimizer import ContextManager

class TestRandomBatch(unittest.TestCase):
    def setUp(self):
        self.mock_model = Mock()
        self.mock_optimizer = Mock()
        self.expected_optimum_position = [[0, 0]]
        self.mock_optimizer.optimize.return_value = self.expected_optimum_position, self.expected_optimum_position
        domain = [{'name': 'var_1', 'type': 'continuous', 'domain': (-5,5), 'dimensionality': 2}]
        self.space = Design_space(domain, None)
        self.mock_optimizer.context_manager = ContextManager(self.space)
        self.ei_acquisition = AcquisitionEI(self.mock_model, self.space, self.mock_optimizer)
        
        self.random_batch = RandomBatch(self.ei_acquisition, 10)
    
    def test_initialize_batch(self):
        x = self.random_batch.initialize_batch()
        
        self.assertEqual(x,self.expected_optimum_position)
        
    def test_get_anchor_points(self):
        anchor_points = self.random_batch.get_anchor_points()
            
        self.assertEqual((50,2),anchor_points.shape)
        self.assertTrue(np.absolute(np.mean(anchor_points)) < 1.5)
        self.assertTrue(np.std(anchor_points) < 4)

    def test_optimize_anchor_points(self):
        self.assertEqual(self.random_batch.optimize_anchor_point(5), 5)
        
    def test_compute_batch_without_duplicate_logic(self):
        points = self.random_batch.compute_batch_without_duplicate_logic()
        
        self.assertEqual((10,2),points.shape)
        self.assertTrue(np.absolute(np.mean(points)) < 2)
        self.assertTrue(np.std(points) < 4)

    def test_compute_batch(self):
        batch = self.random_batch.compute_batch()
    
        self.assertEqual((10,2),batch.shape)
        self.assertTrue(np.absolute(np.mean(batch)) < 2)
        self.assertTrue(np.std(batch) < 3.5)

    def test_zip_and_tuple(self):
        zipped_tuple = self.random_batch.zip_and_tuple(np.array([[1,2],[1,3],[2,2]]))
    
        self.assertEqual(zipped_tuple, (1,2,1,3,2,2))
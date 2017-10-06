import unittest
import numpy as np

from GPyOpt.core.task import CostModel
from GPyOpt.core.task.cost import constant_cost_withGradients

class TestCostModel(unittest.TestCase):
    def test_cost_gp(self):
        cost_model = CostModel(None)
        
        with self.assertRaises(AttributeError):
            time_cost = cost_model._cost_gp(np.array([[3,0],[4,1],[5,1]]))
        
    def test_cost_gp_evaluation_time(self):
        cost_model = CostModel('evaluation_time')
        cost_model.update_cost_model(np.array([[3,0],[4,1],[5,1]]),np.array([4,5,6]))
        time_cost = cost_model._cost_gp(np.array([[3,0],[4,1],[5,1]]))   
        
        self.assertTrue(np.isclose(time_cost,np.array([[4.00000008],[ 5.00000038],[ 5.99999939]])).all())
        
    def test_cost_gp_defined_cost(self):
        cost_model = CostModel(5)
        
        with self.assertRaises(AttributeError):
            time_cost = cost_model._cost_gp(np.array([[3,0],[4,1],[5,1]])) 
    
    def test_cost_withGradients_constant_cost(self):
        cost_model = CostModel(None)
        
        cost, d_cost = cost_model.cost_withGradients(np.array([2,2]))
        
        self.assertTrue(np.isclose(cost,np.array([[1.0],[1.0]])).all())
        self.assertTrue(np.isclose(d_cost,np.array([0.0, 0.0])).all())

    def test_cost_withGradients_evaluation_time(self):
        cost_model = CostModel('evaluation_time')
        cost_model.update_cost_model(np.array([[3,0],[4,1],[5,1]]),np.array([4,5,6]))
        
        cost, d_cost = cost_model.cost_withGradients(np.array([2,2]))
        
        self.assertTrue(np.isclose(cost,np.array([3.52110177])))
        self.assertTrue(np.isclose(d_cost,np.array([0.65617088, 0.08849139]),1e-04).all())
        
    def test_cost_withGradients_user_defined_cost(self):
        def f(x):
            return x*x
            
        cost_model = CostModel(f)
        cost, d_cost = cost_model.cost_withGradients(np.array([2,2]))

        self.assertEqual(cost,4)
        self.assertEqual(d_cost,4)
    
    def test_update_cost_model(self):
        cost_model = CostModel('evaluation_time')

        x = np.array([[3,0],[4,1],[5,1]])
        cost_model.update_cost_model(x,np.array([4,5,6]))
        
        self.assertTrue(cost_model.num_updates == 1)
        self.assertEqual(x.all(), cost_model.cost_model.model.X.all())
        self.assertTrue(np.isclose(cost_model.cost_model.model.Y, np.array([[1.38629436], [1.60943791], [1.79175947]])).all())
        
    def test_update_cost_model_repeat(self):
        cost_model = CostModel('evaluation_time')

        x1 = np.array([[3,0],[4,1],[5,1]])
        cost_model.update_cost_model(x1,np.array([4,5,6]))
        x2 = np.array([[1,1],[3,2],[4,3]])
        cost_model.update_cost_model(x2,np.array([4,2,1]))

        self.assertTrue(cost_model.num_updates == 2)
        self.assertEqual(np.concatenate((x1,x2)).all(), cost_model.cost_model.model.X.all())
        self.assertTrue(np.isclose(cost_model.cost_model.model.Y, np.array([[1.38629436], [1.60943791], [1.79175947],[1.38629436],[0.69314718],[0.0]])).all())

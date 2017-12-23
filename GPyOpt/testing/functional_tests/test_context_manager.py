import unittest
import numpy as np

from GPyOpt.core.task.space import Design_space
from GPyOpt.experiment_design import initial_design
from GPyOpt.optimization.acquisition_optimizer import ContextManager

class TestContextManager(unittest.TestCase):
    def test_context_hadler(self):
        space = [
        {'name': 'var1', 'type': 'continuous', 'domain':(-3,1), 'dimensionality': 3},
        {'name': 'var2', 'type': 'discrete', 'domain': (0,1,2,3)},
        {'name': 'var3', 'type': 'continuous', 'domain':(-5,5)},
        {'name': 'var4', 'type': 'categorical', 'domain': (0, 1)}
        ]

        context = {'var1_1':0.45,'var3':0.52}

        design_space = Design_space(space)
        np.random.seed(666)

        self.context_manager = ContextManager(space = design_space, context = context)

        noncontext_bounds = [(-3, 1), (-3, 1), (0, 3), (0, 1), (0, 1)]
        noncontext_index = [1, 2, 3, 5, 6]
        expanded_vector = np.array([[ 0.45,  0.  ,  0.  ,  0.  ,  0.52,  0.  ,  0.  ]])

        assert np.all(noncontext_bounds ==  self.context_manager.noncontext_bounds )
        assert np.all(noncontext_index  ==  self.context_manager.noncontext_index)
        assert np.all(expanded_vector   ==  self.context_manager._expand_vector(np.array([[0,0,0,0,0]])))

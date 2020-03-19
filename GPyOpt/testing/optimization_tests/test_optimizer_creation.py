import numpy as np
import unittest

from GPyOpt.core.errors import InvalidVariableNameError
from GPyOpt.core.task.space import Design_space
from GPyOpt.optimization.optimizer import choose_optimizer


class TestOptimizerCreation(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestOptimizerCreation, self).__init__(*args, **kwargs)

        self.space = [
            {'name': 'var_1', 'type': 'continuous', 'domain': (-1, 1), 'dimensionality': 1},
            {'name': 'var_2', 'type': 'continuous', 'domain': (-1, 1), 'dimensionality': 1}
        ]
        self.design_space = Design_space(self.space)
        self.f = lambda x: np.sum(np.sin(x))

    def test_invalid_optimizer_name_raises_error(self):
        self.assertRaises(InvalidVariableNameError,
                          choose_optimizer, 'asd', None)

    def test_create_lbfgs_optimizer(self):
        optimizer = choose_optimizer('lbfgs', self.design_space.get_bounds())

        self.assertIsNotNone(optimizer)

    def test_create_direct_optimizer(self):
        optimizer = choose_optimizer('DIRECT', self.design_space.get_bounds())

        self.assertIsNotNone(optimizer)

    def test_create_cma_optimizer(self):
        optimizer = choose_optimizer('CMA', self.design_space.get_bounds())

        self.assertIsNotNone(optimizer)

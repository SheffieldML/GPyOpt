import unittest

from GPyOpt.optimization.optimizer import choose_optimizer
from GPyOpt.core.errors import InvalidVariableNameError


class TestOptimizerCreation(unittest.TestCase):
    """
    Test creation of optimizers
    """

    def test_invalid_optimizer_name_raises_error(self):
        """
        Test error thrown for invalid name
        """

        self.assertRaises(InvalidVariableNameError,
                          choose_optimizer, 'asd', None)

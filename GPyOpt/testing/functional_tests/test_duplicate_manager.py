import unittest
import numpy as np

from GPyOpt.core.task.space import Design_space
from GPyOpt.experiment_design import initial_design
from GPyOpt.util.duplicate_manager import DuplicateManager

class TestDuplicateManager(unittest.TestCase):
    def test_duplicate(self):
        space = [
            {'name': 'var_1', 'type': 'continuous', 'domain':(-3,1), 'dimensionality': 1},
            {'name': 'var_2', 'type': 'discrete', 'domain': (0,1,2,3)},
            {'name': 'var_3', 'type': 'categorical', 'domain': (0, 1)}
        ]
        design_space = Design_space(space)

        np.random.seed(666)

        number_points = 5

        zipped_X = initial_design("random",design_space,number_points)

        d = DuplicateManager(design_space, zipped_X)

        duplicate = np.atleast_2d(zipped_X[0,:].copy())

        assert d.is_zipped_x_duplicate(duplicate)

        assert d.is_unzipped_x_duplicate(design_space.unzip_inputs(duplicate))

        non_duplicate = np.array([[-2.5,  2., 0.]])

        for x in zipped_X:
            assert not np.all(non_duplicate==x)

        assert not d.is_zipped_x_duplicate(non_duplicate)

        assert not d.is_unzipped_x_duplicate(design_space.unzip_inputs(non_duplicate))

    def test_duplicate_with_ignored_and_pending(self):
        space = [
            {'name': 'var_1', 'type': 'continuous', 'domain':(-3,1), 'dimensionality': 1},
            {'name': 'var_2', 'type': 'discrete', 'domain': (0,1,2,3)},
            {'name': 'var_3', 'type': 'categorical', 'domain': (0, 1)}
        ]
        design_space = Design_space(space)

        np.random.seed(666)

        number_points = 5

        zipped_X = initial_design("random",design_space,number_points)
        pending_zipped_X = initial_design("random", design_space, number_points)
        ignored_zipped_X = initial_design("random", design_space, number_points)

        d = DuplicateManager(design_space, zipped_X, pending_zipped_X, ignored_zipped_X)

        duplicate_in_pending_state = np.atleast_2d(pending_zipped_X[0,:].copy())

        assert d.is_zipped_x_duplicate(duplicate_in_pending_state)

        assert d.is_unzipped_x_duplicate(design_space.unzip_inputs(duplicate_in_pending_state))

        duplicate_in_ignored_state = np.atleast_2d(ignored_zipped_X[0,:].copy())

        assert d.is_zipped_x_duplicate(duplicate_in_ignored_state)

        assert d.is_unzipped_x_duplicate(design_space.unzip_inputs(duplicate_in_ignored_state))


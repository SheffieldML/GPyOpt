import numpy as np
import unittest

from GPyOpt.models.input_warped_gpmodel import InputWarpedGPModel
from GPyOpt.core.task.space import Design_space


class TestModels(unittest.TestCase):

    def test_input_warping_indices(self):
        config1 = [{'name': 'var_1', 'type': 'continuous', 'domain':(-3,1), 'dimensionality': 2},
                  {'name': 'var_2', 'type': 'continuous', 'domain':(-3,1), 'dimensionality': 1}]
        warp_ind1 = [0, 1, 2]
        space1 = Design_space(config1)
        m1 = InputWarpedGPModel(space1)
        self.assertEqual(m1.warping_indices, warp_ind1)

        config2 = [{'name': 'var_1', 'type': 'categorical', 'domain': (0,1,2,3)},
                  {'name': 'var_2', 'type': 'continuous', 'domain':(-3,1), 'dimensionality': 1}]
        warp_ind2 = [1]
        space2 = Design_space(config2)
        m2 = InputWarpedGPModel(space2)
        self.assertEqual(m2.warping_indices, warp_ind2)

        config3 = [{'name': 'var_1', 'type': 'categorical', 'domain': (0,1,2,3)},
                  {'name': 'var_2', 'type': 'continuous', 'domain':(-3,1), 'dimensionality': 1}]
        warp_ind3 = [1]
        space3 = Design_space(config3)
        m3 = InputWarpedGPModel(space3)
        self.assertEqual(m3.warping_indices, warp_ind3)

        config4 = [
            {'name': 'var_3', 'type': 'discrete', 'domain': (0,1,2,3)},
            {'name': 'var_3', 'type': 'continuous', 'domain': (2, 4), 'dimensionality': 2},
            {'name': 'var_1', 'type': 'continuous', 'domain':(-3,1), 'dimensionality': 1}
        ]
        warp_ind4 = [0, 1, 2, 3]
        space4 = Design_space(config4)
        m4 = InputWarpedGPModel(space4)
        self.assertEqual(m4.warping_indices, warp_ind4)

        config5 = [
            {'name': 'var_4', 'type': 'bandit', 'domain': np.array([[-2, -1],[0, 1]])}
        ]
        warp_ind5 = []
        space5 = Design_space(config5)
        m5 = InputWarpedGPModel(space5)
        self.assertEqual(m5.warping_indices, warp_ind5)

    def test_input_warping_model(self):
        config1 = [{'name': 'var_1', 'type': 'continuous', 'domain':(-3,1), 'dimensionality': 2},
                   {'name': 'var_2', 'type': 'discrete', 'domain':(-3,1), 'dimensionality': 1}]
        space1 = Design_space(config1)
        m = InputWarpedGPModel(space1)
        np.random.seed(0)
        X = np.random.randn(50, 3)
        Y = np.sum(np.sin(X), 1).reshape(50, 1)
        m._create_model(X, Y)

if __name__ == "__main__":
    print("Running unit tests for GPyOpt Models ...")
    unittest.main()
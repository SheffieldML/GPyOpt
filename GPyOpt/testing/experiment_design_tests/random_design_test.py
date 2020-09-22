import unittest

import numpy as np

from GPyOpt import experiment_design
from GPyOpt.core.task.space import Design_space


class TestRandomDesign(unittest.TestCase):
    def test_random_design_with_all_x_values(self):
        domain = [1, 2]
        space = [{'name': 'var_0',
                  'type': 'discrete',
                  'domain': [1, 2],
                  'dimensionality': 3}]

        # # d_n <= d_n-1 -> d_n - d_n-1 <= 0
        constraints = [{'name': 'const_' + str(i),
                        'constraint': "x[:, " + str(i + 1) + "] - x[:, " + str(i) + "]"}
                       for i in range(0, 2)]  # x1-x0<=0, x2-x1<=0

        all_x_values = np.asarray([[2, 2, 2],
                                   [2, 2, 1],
                                   [2, 1, 1],
                                   [1, 1, 1]])

        design_space = Design_space(space, constraints=constraints, all_x_values=all_x_values)

        X = experiment_design.initial_design(design_name="random", space=design_space, init_points_count=4)
        self.assertEqual(X.shape[0], 4)
        self.assertEqual(X.shape[1], 3)
        self.assertEqual(np.count_nonzero(X == 2), 6)
        self.assertEqual(np.count_nonzero(X == 1), 6)
        X_unique = np.unique(X).tolist()
        self.assertEqual(domain, X_unique)

        X = experiment_design.initial_design(design_name="random", space=design_space, init_points_count=1000)
        self.assertEqual(X.shape[0], 4)
        self.assertEqual(X.shape[1], 3)
        self.assertEqual(np.count_nonzero(X == 2), 6)
        self.assertEqual(np.count_nonzero(X == 1), 6)
        X_unique = np.unique(X).tolist()
        self.assertEqual(domain, X_unique)

        X = experiment_design.initial_design(design_name="random", space=design_space, init_points_count=0)
        self.assertEqual(X.shape[0], 0)
        self.assertEqual(X.shape[1], 3)

        X = experiment_design.initial_design(design_name="random", space=design_space, init_points_count=2)
        self.assertEqual(X.shape[0], 2)
        self.assertEqual(X.shape[1], 3)
        X_unique = np.unique(X).tolist()
        self.assertEqual(domain, X_unique)

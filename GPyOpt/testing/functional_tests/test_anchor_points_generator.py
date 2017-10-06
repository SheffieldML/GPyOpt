import unittest
import numpy as np

from GPyOpt.core.task.space import Design_space
from GPyOpt.util.duplicate_manager import DuplicateManager
from GPyOpt.optimization.anchor_points_generator import ObjectiveAnchorPointsGenerator, ThompsonSamplingAnchorPointsGenerator, RandomAnchorPointsGenerator

tolerance = 1e-8

class TestAnchorPointsGenerator(unittest.TestCase):

    def test_objective_anchor_points_without_duplicate(self):

        space = [
            {'name': 'var_1', 'type': 'continuous', 'domain':(-3,1), 'dimensionality': 1},
            {'name': 'var_2', 'type': 'discrete', 'domain': (0,1,2,3)},
            {'name': 'var_3', 'type': 'categorical', 'domain': (0, 1)}
        ]

        design_space = Design_space(space)

        np.random.seed(666)

        design_type = "random"

        dummy_objective = lambda X : np.sum(X*X, axis=1)

        generator = ObjectiveAnchorPointsGenerator(design_space, design_type, dummy_objective, num_samples=10)

        assert np.all(generator.get_anchor_point_scores(np.arange(3).reshape(3,1)) == np.array([0.0, 1.0, 4.0]))

        assert np.linalg.norm(generator.get(num_anchor=2) - np.array([[-0.02338332, 2., 1., 0.],[ 0.09791782, 2., 1., 0.]])) < tolerance

    def test_objective_anchor_points_with_duplicate(self):

        space = [
            {'name': 'var_1', 'type': 'discrete', 'domain':(-1,2)},
            {'name': 'var_2', 'type': 'discrete', 'domain': (0,1)},
            {'name': 'var_3', 'type': 'categorical', 'domain': (0, 1)}
        ]

        design_space = Design_space(space)

        np.random.seed(666)

        design_type = "random"

        dummy_objective = lambda X : np.sum(X*X, axis=1)

        generator = ObjectiveAnchorPointsGenerator(design_space, design_type, dummy_objective, num_samples=1000)

        initial_points = np.array([[-1, 1, 0],[-1, 1, 1]])

        duplicate_manager = DuplicateManager(design_space, initial_points)

        # There is a total of 2x2x2=8 possible configurations, minus the 2 defined in initial_points
        solution = np.array([[-1.,  0.,  1.,  0.], [-1.,  0.,  0.,  1.], [ 2.,  0.,  1.,  0.], [ 2.,  0.,  0.,  1.], [ 2.,  1.,  1.,  0.], [ 2.,  1.,  0.,  1.]])
        anchor_points = generator.get(num_anchor=6, duplicate_manager=duplicate_manager, unique=True)
        self.assertTrue(np.all(anchor_points == solution))

        all_points = np.vstack((initial_points,design_space.zip_inputs(solution)))
        duplicate_manager_with_all_points = DuplicateManager(design_space, all_points)

        # There aren't any more candidates to generate, hence the exception
        self.assertRaises(Exception, lambda : generator.get(num_anchor=1, duplicate_manager=duplicate_manager_with_all_points, unique=True))

    def test_ts_anchor_points_without_duplicate(self):

        space = [
            {'name': 'var_1', 'type': 'continuous', 'domain':(-3,1), 'dimensionality': 1},
            {'name': 'var_2', 'type': 'discrete', 'domain': (0,1,2,3)},
            {'name': 'var_3', 'type': 'categorical', 'domain': (0, 1)}
        ]

        design_space = Design_space(space)

        np.random.seed(666)

        design_type = "random"

        # We mock a model
        class dummy_model:
            def predict(self,X):
                n = X.shape[0]
                return np.zeros(n), np.ones(n)

        generator = ThompsonSamplingAnchorPointsGenerator(design_space, design_type, dummy_model(), num_samples=10)

        scores = generator.get_anchor_point_scores(np.arange(3).reshape(3,1))

        assert np.linalg.norm(scores - np.array([0.82418808, 0.479966, 1.17346801])) < tolerance

        assert np.linalg.norm(generator.get(num_anchor=2) - np.array([[-2.54856939, 2., 1., 0.],[ 0.09791782, 1., 1., 0.]])) < tolerance

    def test_rand_anchor_points_without_duplicate(self):

        space = [
            {'name': 'var_1', 'type': 'continuous', 'domain':(-3,1), 'dimensionality': 1},
            {'name': 'var_2', 'type': 'discrete', 'domain': (0,1,2,3)},
            {'name': 'var_3', 'type': 'categorical', 'domain': (0, 1)}
        ]

        design_space = Design_space(space)

        np.random.seed(666)

        design_type = "random"

        generator = RandomAnchorPointsGenerator(design_space, design_type, num_samples=10)

        scores = generator.get_anchor_point_scores(np.arange(3).reshape(3,1))

        assert np.linalg.norm(scores - np.array([0., 1., 2.])) < tolerance

        assert np.linalg.norm(generator.get(num_anchor=2) - np.array([[-2.19900984, 0., 0., 1.],[ -0.02338332, 2., 1., 0.]])) < tolerance

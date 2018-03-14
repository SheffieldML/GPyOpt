from __future__ import division

import numpy as np
import unittest
from numpy.testing import assert_allclose

from GPyOpt.core.errors import InvalidConfigError
from GPyOpt.core.task.space import Design_space
from GPyOpt.experiment_design import initial_design
from GPyOpt.util.general import normalize


class TestInitialDesign(unittest.TestCase):
    def setUp(self):
        self.space = [
            {'name': 'var_1', 'type': 'continuous', 'domain':(-3,1), 'dimensionality': 1},
            {'name': 'var_2', 'type': 'discrete', 'domain': (0,1,2,3)},
            {'name': 'var_3', 'type': 'categorical', 'domain': (0,1,2)}
        ]
        self.design_space = Design_space(self.space)

        self.bandit_variable = {'name': 'stations', 'type': 'bandit', 'domain': np.array([[1, 1], [2, 2], [3, 3], [4, 4]])}

    def assert_samples_against_space(self, samples):
        lower_bound_var1 = self.design_space.name_to_variable['var_1'].domain[0]
        upper_bound_var1 = self.design_space.name_to_variable['var_1'].domain[1]
        self.assertTrue((samples[:,0] >= lower_bound_var1).all())
        self.assertTrue((samples[:,0] <= upper_bound_var1).all())

        var2_values = self.design_space.name_to_variable['var_2'].domain
        self.assertTrue(np.in1d(samples[:,1], var2_values).all())

        var3_values = self.design_space.name_to_variable['var_3'].domain
        self.assertTrue(np.in1d(samples[:,2], var3_values).all())

    def test_grid_design(self):
        init_points_count = 3
        samples = initial_design('grid', self.design_space, init_points_count)
        self.assertEqual(len(samples), init_points_count)
        self.assert_samples_against_space(samples)

        init_points_count = 1000
        samples = initial_design('grid', self.design_space, init_points_count)
        self.assertEqual(len(samples), init_points_count)
        self.assert_samples_against_space(samples)

    def test_grid_design_with_multiple_continuous_variables(self):
        self.space.extend([
            {'name': 'var_5', 'type': 'continuous', 'domain':(0,5), 'dimensionality': 2},
            {'name': 'var_6', 'type': 'continuous', 'domain':(-5,5), 'dimensionality': 1}
        ])
        self.design_space = Design_space(self.space)

        init_points_count = 10
        samples = initial_design('grid', self.design_space, init_points_count)
        self.assertEqual(len(samples), 1)

        init_points_count = 100
        samples = initial_design('grid', self.design_space, init_points_count)
        self.assertEqual(len(samples), 3**4)

    def test_random_design(self):
        init_points_count = 10
        samples = initial_design('random', self.design_space, init_points_count)
        self.assertEqual(len(samples), init_points_count)
        self.assert_samples_against_space(samples)

    def test_random_design_with_constraints(self):
        constraints = [{'name': 'const_1', 'constraint': 'x[:,0]**2 - 1'}]
        self.design_space = Design_space(self.space, constraints=constraints)
        initial_points_count = 10

        samples = initial_design('random', self.design_space, initial_points_count)

        self.assert_samples_against_space(samples)
        self.assertTrue((samples[:,0]**2 - 1 < 0).all())

    def test_random_design_with_bandit_only(self):
        space = [self.bandit_variable]
        self.design_space = Design_space(space)
        initial_points_count = 3

        samples = initial_design('random', self.design_space, initial_points_count)

        self.assertEqual(len(samples), initial_points_count)

    def test_nonrandom_designs_with_constrains(self):
        constraints = [{'name': 'const_1', 'constraint': 'x[:,0]**2 - 1'}]
        self.design_space = Design_space(self.space, constraints=constraints)
        initial_points_count = 10

        with self.assertRaises(InvalidConfigError):
            initial_design('grid', self.design_space, initial_points_count)

        with self.assertRaises(InvalidConfigError):
            initial_design('latin', self.design_space, initial_points_count)

        with self.assertRaises(InvalidConfigError):
            initial_design('sobol', self.design_space, initial_points_count)

    def test_latin_design(self):
        init_points_count = 10
        samples = initial_design('latin', self.design_space, init_points_count)
        self.assertEqual(len(samples), init_points_count)
        self.assert_samples_against_space(samples)

    def test_latin_design_with_multiple_continuous_variables(self):
        self.space.extend([
            {'name': 'var_5', 'type': 'continuous', 'domain':(0,5), 'dimensionality': 2},
            {'name': 'var_6', 'type': 'continuous', 'domain':(-5,5), 'dimensionality': 1}
        ])
        self.design_space = Design_space(self.space)

        init_points_count = 10
        samples = initial_design('latin', self.design_space, init_points_count)
        self.assertEqual(len(samples), init_points_count)
        self.assert_samples_against_space(samples)

    def test_sobol_design(self):
        init_points_count = 10
        samples = initial_design('sobol', self.design_space, init_points_count)
        self.assertEqual(len(samples), init_points_count)
        self.assert_samples_against_space(samples)


class TestNormalization(unittest.TestCase):
    """Test the normalize function."""
    def test_call_error(self):
        """Test that 2D arrays raise an error"""
        self.assertRaises(NotImplementedError, normalize, [[1, 2], [3, 4]])

    def test_wrong_type(self):
        """Test whether wrong normalization types raise errors."""
        self.assertRaises(ValueError, normalize, [1], normalization_type='wrong')

    def test_std(self):
        """Test the std normalization."""
        # Zero-std case
        y = np.array([1])
        y_norm = normalize(y, 'stats')
        assert_allclose(y_norm, y - 1)

        y = np.array([0, 0, 0])
        y_norm = normalize(y + 1, 'stats')
        assert_allclose(y_norm, y)

        y = np.array([1, 3])[:, None]
        y_norm = normalize(y, 'stats')
        assert_allclose(y_norm, np.array([[-1], [1]]))

        y = np.arange(5)
        y_norm = normalize(y, 'stats')
        assert_allclose(y_norm, (y - y.mean()) / y.std())

    def test_minmax(self):
        # Zero-std case
        y = np.array([1])
        y_norm = normalize(y, 'maxmin')
        assert_allclose(y_norm, y - 1)

        y = np.array([0, 0, 0])
        y_norm = normalize(y + 1, 'maxmin')
        assert_allclose(y_norm, y)

        y = np.array([1, 3])[:, None]
        y_norm = normalize(y, 'maxmin')
        assert_allclose(y_norm, np.array([[0], [1]]))

        y = np.arange(5)
        y_norm = normalize(y - 1, 'maxmin')
        assert_allclose(y_norm, y / 4)

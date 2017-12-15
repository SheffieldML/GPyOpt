import numpy as np
import unittest

from GPyOpt.core.task.space import Design_space
from GPyOpt.core.task.variables import BanditVariable, DiscreteVariable, CategoricalVariable, ContinuousVariable
from GPyOpt.core.errors import InvalidConfigError

class TestDesignSpace(unittest.TestCase):
    def test_create_bandit_variable(self):
        space = [{'name': 'var_1', 'type': 'bandit', 'domain': np.array([[-1],[0],[1]])}]
        design_space = Design_space(space)

        self.assertEqual(len(design_space.space_expanded), 1)
        self.assertIsInstance(design_space.space_expanded[0], BanditVariable)

    def test_invalid_bandit_config(self):
        space = [
            {'name': 'var_1', 'type': 'continuous', 'domain':(-3,1), 'dimensionality': 2},
            {'name': 'var_3', 'type': 'discrete', 'domain': (0,1,2,3)},
            {'name': 'var_4', 'type': 'categorical', 'domain': (2, 4)},
            {'name': 'var_5', 'type': 'bandit', 'domain': np.array([[-1],[0],[1]])}
        ]

        with self.assertRaises(InvalidConfigError):
            design_space = Design_space(space)

    def test_invalid_mixed_config(self):
        space = [{'name': 'var_1', 'type': 'bandit', 'domain': np.array([[-1, 1], [1]])}]

        with self.assertRaises(InvalidConfigError):
            design_space = Design_space(space)

    def test_create_continuous_variable(self):
        space = [{'name': 'var_1', 'type': 'continuous', 'domain':(-3,1), 'dimensionality': 1}]
        design_space = Design_space(space)

        self.assertEqual(len(design_space.space_expanded), 1)
        self.assertIsInstance(design_space.space_expanded[0], ContinuousVariable)

        space = [{'name': 'var_1', 'type': 'continuous', 'domain':(-3,1), 'dimensionality': 2}]
        design_space = Design_space(space)

        self.assertEqual(len(design_space.space_expanded), 2)
        self.assertTrue(all(isinstance(var, ContinuousVariable) for var in design_space.space_expanded))

    def test_create_discrete_variable(self):
        space = [{'name': 'var_3', 'type': 'discrete', 'domain': (0,1,2,3)}]
        design_space = Design_space(space)

        self.assertEqual(len(design_space.space_expanded), 1)
        self.assertIsInstance(design_space.space_expanded[0], DiscreteVariable)

    def test_create_categorical_variable(self):
        space = [{'name': 'var_3', 'type': 'categorical', 'domain': (0,1,2,3)}]
        design_space = Design_space(space)

        self.assertEqual(len(design_space.space_expanded), 1)
        self.assertIsInstance(design_space.space_expanded[0], CategoricalVariable)

    def test_create_continuous_by_default(self):
        space = [{'domain':(-1,1)}]

        design_space = Design_space(space)

        self.assertEqual(len(design_space.space_expanded), 1)
        self.assertIsInstance(design_space.space_expanded[0], ContinuousVariable)

    def test_domain_missing(self):
        space = [{'name': 'var_1', 'type': 'continuous'}]

        with self.assertRaises(InvalidConfigError):
            Design_space(space)

    def test_dimensionality(self):
        space = [
            {'name': 'var_1', 'type': 'continuous', 'domain':(-3,1), 'dimensionality': 2},
            {'name': 'var_2', 'type': 'discrete', 'domain': (0,1,2,3)},
            {'name': 'var_3', 'type': 'categorical', 'domain': (2, 4)}
        ]

        design_space = Design_space(space)

        self.assertEqual(len(design_space.space_expanded), design_space.dimensionality)


        space = [
            {'name': 'var_1', 'type': 'bandit', 'domain': np.array([[-2, 2],[0, 1],[2, 3]])}
        ]

        design_space = Design_space(space)

        self.assertEqual(len(design_space.space_expanded), 1)
        self.assertEqual(design_space.dimensionality, 2)

    def test_create_constraints(self):
        space = [{'name': 'var_1', 'type': 'continuous', 'domain':(-1,1), 'dimensionality': 2}]
        constraints = [ {'name': 'const_1', 'constraint': 'x[:,0]**2 + x[:,1]**2 - 1'}]

        design_space = Design_space(space, constraints=constraints)

        self.assertEqual(len(design_space.space_expanded), 2)
        self.assertTrue(design_space.has_constraints())

    def test_bounds(self):
        space = [
            {'name': 'var_1', 'type': 'continuous', 'domain':(-3,1), 'dimensionality': 1},
            {'name': 'var_2', 'type': 'discrete', 'domain': (0,1,2,3)},
            {'name': 'var_3', 'type': 'categorical', 'domain': (2, 4)}
        ]

        design_space = Design_space(space)
        bounds = design_space.get_bounds()

        # Countinuous variable bound
        self.assertIn((-3, 1), bounds)
        # Discrete variable bound
        self.assertIn((0, 3), bounds)
        # Categorical variable bound
        self.assertIn((0, 1), bounds)

    def test_bandit_bounds(self):
        space = [{'name': 'var_4', 'type': 'bandit', 'domain': np.array([[-2],[0],[2]])}]

        design_space = Design_space(space)
        bounds = design_space.get_bounds()

        # Bandit variable bound
        self.assertIn((-2, 2), bounds)

    def test_zip_and_unzip(self):
        space = [
            {'name': 'var_1', 'type': 'continuous', 'domain':(-3,1), 'dimensionality': 1},
            {'name': 'var_2', 'type': 'discrete', 'domain': (0,1,2,3), 'dimensionality': 1},
            {'name': 'var_3', 'type': 'categorical', 'domain': (2, 4, 6)}
        ]
        X = np.array([
            [0.0, 1, 2],
            [1.5, 3, 2]
        ])

        design_space = Design_space(space)
        unzipped = design_space.unzip_inputs(X)
        zipped = design_space.zip_inputs(unzipped)

        self.assertTrue(np.array_equal(X, zipped))

    def test_input_dimensions(self):
        space = [
            {'name': 'var_1', 'type': 'continuous', 'domain':(-3,1), 'dimensionality': 1},
            {'name': 'var_2', 'type': 'discrete', 'domain': (0,1,2,3), 'dimensionality': 1},
            {'name': 'var_3', 'type': 'categorical', 'domain': (2, 4, 6)}
        ]

        design_space = Design_space(space)

        self.assertEqual(design_space.input_dim(), 2)



        space[0]['dimensionality'] = 3

        design_space = Design_space(space)

        self.assertEqual(design_space.input_dim(), 4)

    def test_indicator_constraints(self):
        space = [{'name': 'var_1', 'type': 'continuous', 'domain':(-5, 5), 'dimensionality': 1}]
        constraints = [ {'name': 'const_1', 'constraint': 'x[:,0]**2 - 1'}]
        x = np.array([[0], [0.5], [4], [-0.2], [-5]])
        expected_indicies = np.array([[1], [1], [0], [1], [0]])

        design_space = Design_space(space, constraints=constraints)
        I_x = design_space.indicator_constraints(x)

        self.assertTrue(np.array_equal(expected_indicies, I_x))

    def test_invalid_constraint(self):
        space = [{'name': 'var_1', 'type': 'continuous', 'domain':(-5, 5), 'dimensionality': 1}]
        constraints = [{'name': 'const_1', 'constraint': 'x[:,20]**2 - 1'}]
        x = np.array([[0]])

        design_space = Design_space(space, constraints=constraints)

        with self.assertRaises(Exception):
            design_space.indicator_constraints(x)

    def test_subspace(self):
        space = [
            {'name': 'var_1', 'type': 'continuous', 'domain':(-1,1), 'dimensionality': 2},
            {'name': 'var_2', 'type': 'categorical', 'domain': ('r', 'g', 'b')},
            {'name': 'var_3', 'type': 'discrete', 'domain': (0,1,2,3)}
        ]
        dims = [0, 2, 5]
        design_space = Design_space(space)

        subspace = design_space.get_subspace(dims)

        self.assertEqual(len(subspace), 3)
        self.assertTrue(any(v for v in subspace if isinstance(v, ContinuousVariable)))
        self.assertTrue(any(v for v in subspace if isinstance(v, DiscreteVariable)))
        self.assertTrue(any(v for v in subspace if isinstance(v, CategoricalVariable)))

    def test_bandit(self):
        X =     np.array([
                [0, -2, -1],
                [ 0,  0,  1],
                [ 1, -2, -1],
                [ 1,  0,  1],
                [ 3, -2, -1],
                [ 3,  0, 1]])

        space = [{'name': 'var', 'type': 'bandit', 'domain':X}]

        design_space = Design_space(space)

        self.assertTrue(design_space._has_bandit())
        self.assertTrue(design_space.unzip_inputs(X).all()==X.all())
        self.assertTrue(design_space.zip_inputs(X).all()==X.all())

    def test_variable_names(self):
        space = [
            {'name': 'var1', 'type': 'continuous', 'domain':(-1,1), 'dimensionality': 2},
            {'name': 'var2', 'type': 'categorical', 'domain': ('r', 'g', 'b'), 'dimensionality': 3},
            {'name': 'var3', 'type': 'discrete', 'domain': (0,1,2,3), 'dimensionality': 2},
            {'name': 'var4', 'type': 'continuous', 'domain':(-1,1)},
            {'name': 'var5', 'type': 'continuous', 'domain':(-1,1), 'dimensionality': 1}
        ]

        design_space = Design_space(space)

        var_names = ['var1_1','var1_2','var2_1','var2_2','var2_3','var3_1','var3_2','var4','var5']

        k = 0
        for variable in design_space.space_expanded:
            self.assertTrue(variable.name == var_names[k])
            k+=1

    # Generic unit test runner for round_optimum tests
    # TODO: Refactor to use subtests once we deprecate Python 2: https://stackoverflow.com/a/29384495
    def assert_round_optimum(self, space, test_cases):
        design_space = Design_space(space)
        for test_case in test_cases:
            rounded = design_space.round_optimum(np.array(test_case['in']))
            self.assertTrue(np.array_equal(rounded, np.array(test_case['out'])))

    def test_round_optimum_continuous(self):
        space = [
            {'name': 'var1', 'type': 'continuous', 'domain': [-1, 1], 'dimensionality': 1}
        ]
        test_cases = [
            {'in': [[1]], 'out': [[1]]},
            {'in': [[-3]], 'out': [[-1]]},
            {'in': [[4]], 'out': [[1]]},
            {'in': [[0.5]], 'out': [[0.5]]}
        ]
        self.assert_round_optimum(space, test_cases)

        space = [
            {'name': 'var1', 'type': 'continuous', 'domain': [-1, 1], 'dimensionality': 2}
        ]
        test_cases = [{'in': [[0.5, -2]], 'out': [[0.5, -1]]}]
        self.assert_round_optimum(space, test_cases)

    def test_round_optimum_discrete(self):
        space = [
            {'name': 'var1', 'type': 'discrete', 'domain': (0,1,2,3,6), 'dimensionality': 1}
        ]
        test_cases = [
            {'in': [[1]], 'out': [[1]]},
            {'in': [[-3]], 'out': [[0]]},
            {'in': [[4]], 'out': [[3]]},
            {'in': [[5]], 'out': [[6]]},
            {'in': [[8]], 'out': [[6]]}
        ]
        self.assert_round_optimum(space, test_cases)

        space = [
            {'name': 'var1', 'type': 'discrete', 'domain': (0,1,2,3,6), 'dimensionality': 2}
        ]
        test_cases = [{'in': [[1, 5]], 'out': [[1, 6]]}]
        self.assert_round_optimum(space, test_cases)

    def test_round_optimum_categorical(self):
        space = [
            {'name': 'var1', 'type': 'categorical', 'domain': (0,1,2,3), 'dimensionality': 1}
        ]
        test_cases = [
            {'in': [[1, 0, 0, 0]], 'out': [[1, 0, 0, 0]]},
            {'in': [[1, 0, 1, 0]], 'out': [[1, 0, 0, 0]]},
            {'in': [[1, 2, 3, 1]], 'out': [[0, 0, 1, 0]]},
            {'in': [[-1, -2, -3, 0]], 'out': [[0, 0, 0, 1]]}
        ]
        self.assert_round_optimum(space, test_cases)

        space = [
            {'name': 'var1', 'type': 'categorical', 'domain': (0,1,2,3), 'dimensionality': 2}
        ]
        test_cases = [
            {'in': [[1, 0, 0, 0, 1, 1, 1, 1]], 'out': [[1, 0, 0, 0, 1, 0, 0, 0]]}
        ]
        self.assert_round_optimum(space, test_cases)

    def test_round_optimum_mixed_domain(self):
        space = [
            {'name': 'var1', 'type': 'continuous', 'domain':(-1,1), 'dimensionality': 1},
            {'name': 'var2', 'type': 'categorical', 'domain': (0, 1, 2), 'dimensionality': 1},
            {'name': 'var3', 'type': 'discrete', 'domain': (0, 3, 5), 'dimensionality': 1},
        ]
        test_cases = [
            {'in': [[0, 0, 1, 1, 2]], 'out': [[0, 0, 1, 0, 3]]}
        ]
        self.assert_round_optimum(space, test_cases)

    def test_round_optimum_bandit(self):
        space = [
            {'name': 'var1', 'type': 'bandit', 'domain': np.array([[-2, 2],[0, 1],[2, 3]])}
        ]
        test_cases = [
            {'in': [[0, -1]], 'out': [[0, 1]]},
            {'in': [[2, 2]], 'out': [[2, 3]]},
            {'in': [[1, 1]], 'out': [[0, 1]]},
            {'in': [[100, 200]], 'out': [[2, 3]]},
            {'in': [[-3, 2]], 'out': [[-2, 2]]}
        ]
        self.assert_round_optimum(space, test_cases)

    def test_round_optimum_shapes(self):
        space = [{'name': 'var1', 'type': 'continuous', 'domain':(-1,1), 'dimensionality': 1}]

        with self.assertRaises(ValueError):
            design_space = Design_space(space)
            design_space.round_optimum([[[0.0]]])

        with self.assertRaises(ValueError):
            design_space = Design_space(space)
            design_space.round_optimum(np.array([[[0.0]]]))

        with self.assertRaises(ValueError):
            design_space = Design_space(space)
            design_space.round_optimum(np.array([[0.0], [2.0]]))

        # Next couple of tests are intentionally very simple
        # as they just verify that exception is not thrown for the given input shape
        design_space = Design_space(space)

        rounded = design_space.round_optimum([0.0])
        self.assertEqual(rounded[0], 0.0)

        rounded = design_space.round_optimum(np.array([0.0]))
        self.assertEqual(rounded[0], 0.0)

        rounded = design_space.round_optimum([[0.0]])
        self.assertEqual(rounded[0], 0.0)

        rounded = design_space.round_optimum(np.array([[0.0]]))
        self.assertEqual(rounded[0], 0.0)

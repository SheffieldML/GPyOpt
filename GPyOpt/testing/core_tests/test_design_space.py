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
            {'name': 'var_3', 'type': 'discrete', 'domain': (0,1,2,3)},
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
        constraints = [ {'name': 'const_1', 'constrain': 'x[:,0]**2 + x[:,1]**2 - 1'}]

        design_space = Design_space(space, constraints=constraints)

        self.assertEqual(len(design_space.space_expanded), 2)
        self.assertTrue(design_space.has_constraints())

    def test_bounds(self):
        space = [
            {'name': 'var_1', 'type': 'continuous', 'domain':(-3,1), 'dimensionality': 1},
            {'name': 'var_3', 'type': 'discrete', 'domain': (0,1,2,3)},
            {'name': 'var_3', 'type': 'categorical', 'domain': (2, 4)},
            {'name': 'var_4', 'type': 'bandit', 'domain': np.array([[-2],[0],[2]])}
        ]

        design_space = Design_space(space)
        bounds = design_space.get_bounds()

        # Countinuous variable bound
        self.assertIn((-3, 1), bounds)
        # Discrete variable bound
        self.assertIn((0, 3), bounds)
        # Bandit variable bound
        self.assertIn((-2, 2), bounds)
        # Categorical variable bound
        self.assertIn((0, 1), bounds)

    def test_zip_and_unzip(self):
        space = [
            {'name': 'var_1', 'type': 'continuous', 'domain':(-3,1), 'dimensionality': 1},
            {'name': 'var_2', 'type': 'discrete', 'domain': (0,1,2,3), 'dimensionality': 1},
            {'name': 'var_3', 'type': 'categorical', 'domain': (2, 4, 6)},
            {'name': 'var_4', 'type': 'bandit', 'domain': np.array([[-2],[0],[2]])}
        ]
        X = np.array([
            [0.0, 1, 2, -2],
            [1.5, 3, 2, 2]
        ])

        design_space = Design_space(space)
        unzipped = design_space.unzip_inputs(X)
        zipped = design_space.zip_inputs(unzipped)

        self.assertTrue(np.array_equal(X, zipped))

    def test_input_dimensions(self):
        space = [
            {'name': 'var_1', 'type': 'continuous', 'domain':(-3,1), 'dimensionality': 1},
            {'name': 'var_2', 'type': 'discrete', 'domain': (0,1,2,3), 'dimensionality': 1},
            {'name': 'var_3', 'type': 'categorical', 'domain': (2, 4, 6)},
            {'name': 'var_4', 'type': 'bandit', 'domain': np.array([[-2],[0],[2]])}
        ]

        design_space = Design_space(space)

        self.assertEqual(design_space.input_dim(), 2)



        space[0]['dimensionality'] = 3

        design_space = Design_space(space)

        self.assertEqual(design_space.input_dim(), 4)

    def test_indicator_constraints(self):
        space = [{'name': 'var_1', 'type': 'continuous', 'domain':(-5, 5), 'dimensionality': 1}]
        constraints = [ {'name': 'const_1', 'constrain': 'x[:,0]**2 - 1'}]
        x = np.array([[0], [0.5], [4], [-0.2], [-5]])
        expected_indicies = np.array([[1], [1], [0], [1], [0]])

        design_space = Design_space(space, constraints=constraints)
        I_x = design_space.indicator_constraints(x)

        self.assertTrue(np.array_equal(expected_indicies, I_x))

    def test_invalid_constraint(self):
        space = [{'name': 'var_1', 'type': 'continuous', 'domain':(-5, 5), 'dimensionality': 1}]
        constraints = [{'name': 'const_1', 'constrain': 'x[:,20]**2 - 1'}]
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

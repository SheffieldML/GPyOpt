import unittest
import numpy as np

import GPyOpt
from GPyOpt.methods import BayesianOptimization

class TestBayesianOptimization(unittest.TestCase):
    def test_next_locations_pending(self):
        func = GPyOpt.objective_examples.experiments1d.forrester()
        domain =[{'name': 'var1', 'type': 'continuous', 'domain': (0,1)}]
        X_init = np.array([[0.0],[0.5],[1.0]])
        Y_init = func.f(X_init)

        np.random.seed(1)
        bo_no_pending = BayesianOptimization(f = None, domain = domain, X = X_init, Y = Y_init)
        x_no_pending = bo_no_pending.suggest_next_locations()

        np.random.seed(1)
        bo_pending = BayesianOptimization(f = None, domain = domain, X = X_init, Y = Y_init, de_duplication = True)
        x_pending = bo_pending.suggest_next_locations(pending_X = x_no_pending)

        self.assertFalse(np.isclose(x_pending, x_no_pending))

    def test_next_locations_ignored(self):
        func = GPyOpt.objective_examples.experiments1d.forrester()
        domain =[{'name': 'var1', 'type': 'continuous', 'domain': (0,1)}]
        X_init = np.array([[0.0],[0.5],[1.0]])
        Y_init = func.f(X_init)

        np.random.seed(1)
        bo_no_ignored = BayesianOptimization(f = None, domain = domain, X = X_init, Y = Y_init)
        x_no_ignored = bo_no_ignored.suggest_next_locations()

        np.random.seed(1)
        bo_ignored = BayesianOptimization(f = None, domain = domain, X = X_init, Y = Y_init, de_duplication = True)
        x_ignored = bo_ignored.suggest_next_locations(ignored_X = x_no_ignored)

        self.assertFalse(np.isclose(x_ignored, x_no_ignored))

    def test_one_initial_data_point(self):
        """Make sure BO still works with only one initial data point."""
        bounds = [{'name': 'var_1', 'type': 'continuous', 'domain': (-1, 1)}]
        # Remove hyperparameter optimization, not needed here
        model = GPyOpt.models.GPModel(max_iters=0)
        opt = BayesianOptimization(lambda x: x, bounds, model=model, initial_design_numdata=1)

        # Make sure run_optimization works
        opt.run_optimization(max_iter=1)
        assert len(opt.Y) > 1

    def test_infinite_distance_last_evaluations(self):
        # Optimization with a single data point will have the distance between last evaluations go to infinity
        # This will not be interpreted as a converged state and continue as expected
        domain = [{'name': 'x', 'type': 'continuous', 'domain': (-10,10)}]

        # one initial data point found randomly
        bo = GPyOpt.methods.BayesianOptimization(lambda x: x*x, domain=domain,
                                                 X=None, Y=None,
                                                 initial_design_numdata=1,
                                                 initial_design_type='random')
        bo.run_optimization(max_iter=3)
        assert len(bo.X) == 4
        assert len(bo.Y) == 4

        # one initial data point given by the user
        bo2 = GPyOpt.methods.BayesianOptimization(lambda x: x*x, domain=domain,
                                                  X=np.array([[1]]), Y=np.array([[1]]),
                                                  initial_design_numdata=0)
        bo2.run_optimization(max_iter=2)
        assert len(bo2.X) == 3
        assert len(bo2.Y) == 3

    def test_normalization(self):
        """Make sure normalization works with wrong model."""
        np.random.seed(1)

        bounds = [{'name': 'var_1', 'type': 'continuous', 'domain': (1, 2)}]
        # We get a first measurement at 1, but the model is so harshly
        # violated that we will just continue to get measurements at 1.
        # The minimum is at 2.
        x0 = np.array([[1]])
        f = lambda x: -1000 * x

        # Remove hyperparameter optimization, not needed here
        model = GPyOpt.models.GPModel(max_iters=0)

        opt = BayesianOptimization(f, bounds, X=x0, model=model, normalize_Y=True)
        opt.run_optimization(max_iter=1)

        # Make sure that we did not sample the same point again
        assert np.linalg.norm(opt.X[0] - opt.X[1]) > 1e-2

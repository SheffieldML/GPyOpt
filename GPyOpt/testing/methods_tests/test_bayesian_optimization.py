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
        opt = BayesianOptimization(lambda x: x, bounds, initial_design_numdata=1)

        # Make sure run_optimization works
        opt.run_optimization(max_iter=1)
        assert len(opt.Y) > 1

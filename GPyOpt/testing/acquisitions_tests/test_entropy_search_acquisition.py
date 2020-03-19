import unittest
from mock import Mock

import numpy as np
from numpy.testing import assert_allclose

import GPy

from GPyOpt.acquisitions import AcquisitionEntropySearch
from GPyOpt.core.task.space import Design_space
from GPyOpt.models import GPModel

from GPyOpt.util.mcmc_sampler import McmcSampler, AffineInvariantEnsembleSampler
from GPyOpt.experiment_design import initial_design

class MockSampler(McmcSampler):
    def __init__(self, space):
        self.space = space

    def get_samples(self, n_samples, log_p_function, burn_in_steps=50):
        samples = initial_design('latin', self.space, n_samples)
        samples_log = np.array([[i] for i in range(n_samples)])

        return samples, samples_log


class TestEntropySearchAcquisition(unittest.TestCase):
    def setUp(self):
        np.random.seed(1)

        X = np.array([[-1.5, -1], [1, 1.5], [3, 3]])
        y = 2 * -np.array([[-0.1], [.3], [.9]])
        bounds = [(-5, 5)]
        input_dim = X.shape[1]
        kern = GPy.kern.RBF(input_dim, variance=1., lengthscale=1.)
        self.model = GPModel(kern, noise_var=0.0, max_iters=0, optimize_restarts=0)
        self.model.updateModel(X, y, None, None)
        domain = [{'name': 'var_1', 'type': 'continuous', 'domain': bounds[0], 'dimensionality': 2}]
        self.space = Design_space(domain)

        self.mock_optimizer = Mock()

    def test_acquisition_function(self):
        es = AcquisitionEntropySearch(self.model, self.space, MockSampler(self.space))
        acquisition_value = es.acquisition_function(np.array([[1, 1]]))

        assert_allclose(acquisition_value, np.array([[-20.587977]]), 1e-5)

    def test_optimize(self):
        expected_optimum_position = [[0, 0]]
        self.mock_optimizer.optimize.return_value = expected_optimum_position
        es = AcquisitionEntropySearch(self.model, self.space, MockSampler(self.space), optimizer=self.mock_optimizer)

        optimum_position = es.optimize()

        assert optimum_position == expected_optimum_position

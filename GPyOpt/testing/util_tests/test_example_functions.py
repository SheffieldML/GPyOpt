import unittest
import numpy as np

from GPyOpt.objective_examples.experiments1d import forrester
from GPyOpt.objective_examples.experiments2d import rosenbrock, beale,\
    dropwave, cosines, branin, goldstein, sixhumpcamel, mccormick, powers,\
    eggholder
from GPyOpt.objective_examples.experimentsNd import alpine1, alpine2, ackley


class TestExampleFunctions(unittest.TestCase):
    '''
    This file provides a test for every example function given in
    GPyOpt.objective_examples.*

    Essentially, we are just checking that the specified minimizer x_min
    actually evaluates to the specified function minimum.

    Note that we do not search globally for the true minimum, we only
    assert that, e.g. F = forrester(); F.f(F.min) == F.fmin.
    '''
    def setUp(self):
        return

    def tearDown(self):
        return

    def _check_minimizer(self, fcls):
        '''Checks that the function wrapped by the class fcls correctly
        states it's minimum value and location'''
        xmin = np.atleast_2d(fcls.min)
        fmin = fcls.fmin
        f_xmin = fcls.f(xmin)
        assert np.allclose(fmin, f_xmin, 1e-3, 1e-3),\
            'Incorrect minimizer! f(x_min) = {0}, but fmin = {1}'.format(
                f_xmin, fmin)
        for i in range(xmin.shape[0]):
            xmin_i = xmin[i]
            for j, xmin_ij in enumerate(xmin_i):
                assert xmin_ij >= fcls.bounds[j][0], 'minimizer not in bounds!'
                assert xmin_ij <= fcls.bounds[j][1], 'minimizer not in bounds!'
        return

    def _evaluate_1d(self, fcls):
        '''Evaluates a 1d function wrapped by fcls on it's domain'''
        assert fcls.input_dim == 1, 'Function is not 1D!'
        x = np.arange(*fcls.bounds[0], 0.01)
        fx = fcls.f(x)  # Evaluate the function and let errors through
        return fx

    def _evaluate_2d(self, fcls):
        '''Evaluates a 2d function wrapped by fcls on it's domain'''
        assert fcls.input_dim == 2, 'Function is not 2D!'
        x = np.arange(*fcls.bounds[0], 0.01)
        if len(x) > 1000:  # To ensure the meshgrid doesn't explode memory
            x = np.linspace(*fcls.bounds[0], 1000)

        y = np.arange(*fcls.bounds[1], 0.01)
        if len(y) > 1000:
            y = np.linspace(*fcls.bounds[1], 1000)

        # This produces a N x 2 vector from the cartesian product
        # of x and y https://stackoverflow.com/questions/11144513/
        X = np.dstack(np.meshgrid(x, y)).reshape(-1, 2)
        fx = fcls.f(X)
        return fx

    def test_experiments1d_forrester(self):
        fcls = forrester()
        self._check_minimizer(fcls)
        self._evaluate_1d(fcls)
        return

    def test_experiments2d_rosenbrock(self):
        fcls = rosenbrock()
        self._check_minimizer(fcls)
        self._evaluate_2d(fcls)
        return

    def test_experiments2d_beale(self):
        fcls = beale()
        self._check_minimizer(fcls)
        self._evaluate_2d(fcls)
        return

    def test_experiments2d_dropwave(self):
        fcls = dropwave()
        self._check_minimizer(fcls)
        self._evaluate_2d(fcls)
        return

    def test_experiments2d_cosines(self):
        fcls = cosines()
        self._check_minimizer(fcls)
        self._evaluate_2d(fcls)
        return

    def test_experiments2d_branin(self):
        fcls = branin()
        self._check_minimizer(fcls)
        self._evaluate_2d(fcls)
        return

    def test_experiments2d_goldstein(self):
        fcls = goldstein()
        self._check_minimizer(fcls)
        self._evaluate_2d(fcls)
        return

    def test_experiments2d_sixhumpcamel(self):
        fcls = sixhumpcamel()
        self._check_minimizer(fcls)
        self._evaluate_2d(fcls)
        return

    def test_experiments2d_mccormick(self):
        fcls = mccormick()
        self._check_minimizer(fcls)
        self._evaluate_2d(fcls)
        return

    def test_experiments2d_powers(self):
        fcls = powers()
        self._check_minimizer(fcls)
        self._evaluate_2d(fcls)
        return

    def test_experiments2d_eggholder(self):
        fcls = eggholder()
        self._check_minimizer(fcls)
        self._evaluate_2d(fcls)
        return

    def test_experimentsNd_alpine1(self):
        fcls = alpine1(2)
        self._check_minimizer(fcls)
        self._evaluate_2d(fcls)
        return

    def test_experimentsNd_alpine2(self):
        fcls = alpine2(2)
        self._check_minimizer(fcls)
        self._evaluate_2d(fcls)
        return

    def test_experimentsNd_ackley(self):
        fcls = ackley(2)
        self._check_minimizer(fcls)
        self._evaluate_2d(fcls)
        return

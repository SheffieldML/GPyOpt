import unittest

from GPyOpt.objective_examples.experiments1d import forrester
from GPyOpt.objective_examples.experiments2d import rosenbrock, beale,\
    dropwave, cosines, branin, goldstein, sixhumpcamel, mccormick, powers,\
    eggholder
from GPyOpt.objective_examples.experimentsNd import alpine1, alpine2, gSobol,\
    ackley


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
        assert fcls.f(fcls.min) == fcls.fmin, 'Incorrect minimizer!'
        return

    def test_experiments1d_forrester(self):
        fcls = forrester()
        self._check_minimizer(fcls)
        return

    def test_experiments2d_rosenbrock(self):
        fcls = rosenbrock()
        self._check_minimizer(fcls)
        return

    def test_experiments2d_beale(self):
        fcls = beale()
        self._check_minimizer(fcls)
        return

    def test_experiments2d_dropwave(self):
        fcls = dropwave()
        self._check_minimizer(fcls)
        return

    def test_experiments2d_cosines(self):
        fcls = cosines()
        self._check_minimizer(fcls)
        return

    def test_experiments2d_branin(self):
        fcls = branin()
        self._check_minimizer(fcls)
        return

    def test_experiments2d_goldstein(self):
        fcls = goldstein()
        self._check_minimizer(fcls)
        return

    def test_experiments2d_sixhumpcamel(self):
        fcls = sixhumpcamel()
        self._check_minimizer(fcls)
        return

    def test_experiments2d_mccormick(self):
        fcls = mccormick()
        self._check_minimizer(fcls)
        return

    def test_experiments2d_powers(self):
        fcls = powers()
        self._check_minimizer(fcls)
        return

    def test_experiments2d_eggholder(self):
        fcls = eggholder()
        self._check_minimizer(fcls)
        return

    def test_experimentsNd_alpine1(self):
        fcls = alpine1()
        self._check_minimizer(fcls)
        return

    def test_experimentsNd_alpine2(self):
        fcls = alpine2()
        self._check_minimizer(fcls)
        return

    def test_experimentsNd_gSobol(self):
        fcls = gSobol()
        self._check_minimizer(fcls)
        return

    def test_experimentsNd_ackley(self):
        fcls = ackley()
        self._check_minimizer(fcls)
        return

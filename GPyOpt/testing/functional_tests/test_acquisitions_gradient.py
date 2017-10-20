# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np

import GPy
from GPy.models.gradient_checker import GradientChecker

import GPyOpt
from GPyOpt.util.general import samples_multidimensional_uniform
from GPyOpt.acquisitions import AcquisitionEI, AcquisitionMPI, AcquisitionLCB

from mock import Mock
import unittest


class acquisition_for_test():
    '''
    Class to run the unit test for the gradients of the acquisitions
    '''
    def __init__(self,gpyopt_acuq):
        self.gpyopt_acuq = gpyopt_acuq

    def acquisition_function(self,x):
        return self.gpyopt_acuq.acquisition_function_withGradients(x)[0]

    def d_acquisition_function(self,x):
        return self.gpyopt_acuq.acquisition_function_withGradients(x)[1]


class TestAcquisitionsGradients(unittest.TestCase):
    '''
    Unittest for the gradients of the available acquisition functions
    '''

    def setUp(self):
        np.random.seed(10)
        self.tolerance = 0.05  # Tolerance for difference between true and approximated gradients
        objective = GPyOpt.objective_examples.experiments1d.forrester()
        self.feasible_region =  GPyOpt.Design_space(space = [{'name': 'var_1', 'type': 'continuous', 'domain': objective.bounds[0]}])
        n_inital_design = 10
        X = samples_multidimensional_uniform(objective.bounds,n_inital_design)
        Y = objective.f(X)
        self.X_test = samples_multidimensional_uniform(objective.bounds,n_inital_design)

        self.model = Mock()
        self.model.get_fmin.return_value = 0.0
        self.model.predict_withGradients.return_value = np.zeros(X.shape), np.zeros(Y.shape), np.zeros(X.shape), np.zeros(X.shape)

    def test_ChecKGrads_EI(self):
        acquisition_ei = acquisition_for_test(AcquisitionEI(self.model, self.feasible_region))
        grad_ei = GradientChecker(acquisition_ei.acquisition_function, acquisition_ei.d_acquisition_function, self.X_test)
        self.assertTrue(grad_ei.checkgrad(tolerance=self.tolerance))


    def test_ChecKGrads_MPI(self):
        acquisition_mpi = acquisition_for_test(AcquisitionMPI(self.model, self.feasible_region))
        grad_mpi = GradientChecker(acquisition_mpi.acquisition_function, acquisition_mpi.d_acquisition_function, self.X_test)
        self.assertTrue(grad_mpi.checkgrad(tolerance=self.tolerance))


    def test_ChecKGrads_LCB(self):
        acquisition_lcb = acquisition_for_test(AcquisitionLCB(self.model, self.feasible_region))
        grad_lcb = GradientChecker(acquisition_lcb.acquisition_function, acquisition_lcb.d_acquisition_function, self.X_test)
        self.assertTrue(grad_lcb.checkgrad(tolerance=self.tolerance))


if __name__=='main':
    unittest.main()

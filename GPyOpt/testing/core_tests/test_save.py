import unittest
import numpy as np
from GPyOpt.methods import BayesianOptimization
from GPyOpt.models import GPModel
import GPy
import tempfile
import os


class TestSaveModel(unittest.TestCase):
    """
    Tests to check primarily that saving a BO model results in a file without errors.
    """

    def setUp(self):
        self.f_1d = lambda x: (6*x-2)**2*np.sin(12*x-4)
        self.f_2d = lambda x: (6*x[:,0]-2)**2*np.sin(12*x[:,1]-4)
        self.domain_1d = [{'name': 'var_1', 'type': 'continuous', 'domain': (0,1), 'dimensionality': 1}]
        self.domain_2d = [{'name': 'var_1', 'type': 'continuous', 'domain': (0,1)}, {'name': 'var_2', 'type': 'continuous', 'domain': (0,1)}]

        self.outfile_path = tempfile.mkstemp()[1]
        # Need to delete the file afterwards - no matter what
        self.addCleanup(os.remove, self.outfile_path)

    def check_output_model_file(self, contained_strings):
        """ Rudimentary test to check the file contents contain something sensible - can add to this but don't want to make it too implementation specific """
        contents = open(self.outfile_path).read()
        for substring in contained_strings:
            self.assertTrue(substring in contents)

    def test_save_gp_default_no_iters(self):
        myBopt = BayesianOptimization(f=self.f_2d, domain=self.domain_2d)
        # Exception should be raised as no iterations have been carried out yet
        self.assertRaises(ValueError, lambda: myBopt.save_models(self.outfile_path))

    def test_save_gp_no_filename(self):
        myBopt = BayesianOptimization(f=self.f_2d, domain=self.domain_2d)
        myBopt.run_optimization(max_iter=1, verbosity=False)
        # Need to at least pass in filename or buffer
        self.assertRaises(TypeError, lambda: myBopt.save_models())

    def test_save_gp_default(self):
        myBopt = BayesianOptimization(f=self.f_2d, domain=self.domain_2d)
        myBopt.run_optimization(max_iter=1, verbosity=False)
        myBopt.save_models(self.outfile_path)
        self.check_output_model_file(['Iteration'])

    def test_save_gp_2d(self):
        k = GPy.kern.Matern52(input_dim=2)
        m = GPModel(kernel=k)
        myBopt = BayesianOptimization(f=self.f_2d, domain=self.domain_2d, model=m)
        myBopt.run_optimization(max_iter=1, verbosity=False)
        myBopt.save_models(self.outfile_path)
        self.check_output_model_file(['Iteration'])

    def test_save_gp_2d_ard(self):
        """
        This was previously an edge-case, when some parameters were vectors, the naming of the columns was incorrect
        """
        k = GPy.kern.Matern52(input_dim=2, ARD=True)
        m = GPModel(kernel=k)
        myBopt = BayesianOptimization(f=self.f_2d, domain=self.domain_2d, model=m)
        myBopt.run_optimization(max_iter=1, verbosity=False)
        myBopt.save_models(self.outfile_path)
        self.check_output_model_file(['Iteration'])

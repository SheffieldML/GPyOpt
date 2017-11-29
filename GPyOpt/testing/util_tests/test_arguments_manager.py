import unittest
import numpy as np

from GPyOpt.util.arguments_manager import ArgumentsManager
from GPyOpt.core.task.space import Design_space
from GPyOpt.models.gpmodel import GPModel_MCMC, GPModel
from GPyOpt.models.rfmodel import RFModel
from GPyOpt.models.input_warped_gpmodel import InputWarpedGPModel
from GPyOpt.models.warpedgpmodel import WarpedGPModel
from GPyOpt.core.task.cost import CostModel
from GPyOpt.acquisitions import AcquisitionEI, AcquisitionLCB
from GPyOpt.optimization.acquisition_optimizer import AcquisitionOptimizer
from GPyOpt.core.evaluators import LocalPenalization


class TestArgumentsManager(unittest.TestCase):
    '''
    The test just checks that the arguments are passed and handled, not that the model
    or the acquisition does the right thing with them.
    '''
    def setUp(self):
        kwargs  = {'n_samples':1000,
                    'n_burnin':100,
                    'subsample_interval':5,
                    'step_size': 1e-1,
                    'leapfrog_steps':20,
                    'optimize_restarts':10,
                    'num_inducing':15,
                    'acquisition_transformation':'softplus',
                    'acquisition_jitter':0.02,
                    'acquisition_weight':2.5,
                    'acquisition_transformation':'softplus'
                    }

        ## --- Defaults for some of the tests
        self.space = Design_space(space =[{'name': 'var1', 'type': 'continuous', 'domain': (-10,10),'dimensionality': 2}])
        self.cost = CostModel(None)
        self.arguments_manager = ArgumentsManager(kwargs)
        self.model = self.arguments_manager.model_creator(model_type = 'GP', exact_feval= True, space = self.space)
        self.acquisition_optimizer = AcquisitionOptimizer(self.space)
        self.acquisition = self.arguments_manager.acquisition_creator('EI', self.model, self.space, self.acquisition_optimizer, self.cost)


    def test_model_gp_mcmc_arguments(self):
        '''
        Testing the arguments of the GP model with MCMC
        '''
        model_type = 'GP_MCMC'
        exact_feval = True
        created_model = self.arguments_manager.model_creator(model_type, exact_feval, self.space)
        self.assertTrue(isinstance(created_model, GPModel_MCMC))
        self.assertEquals(created_model.n_samples, 1000)
        self.assertEquals(created_model.n_burnin, 100)
        self.assertEquals(created_model.subsample_interval, 5)
        self.assertEquals(created_model.step_size, 1e-1)
        self.assertEquals(created_model.leapfrog_steps, 20)

    def test_model_sparse_gp_arguments(self):
        '''
        Testing the arguments of the standard GP model
        '''
        model_type = 'sparseGP'
        exact_feval = True
        created_model = self.arguments_manager.model_creator(model_type, exact_feval, self.space)
        self.assertTrue(isinstance(created_model, GPModel))
        self.assertEquals(created_model.optimize_restarts,10)
        self.assertEquals(created_model.num_inducing,15)

    def test_model_rf_arguments(self):
        '''
        Testing the arguments of the Random Forrest
        '''
        model_type = 'RF'
        exact_feval = True
        created_model = self.arguments_manager.model_creator(model_type, exact_feval, self.space)
        self.assertTrue(isinstance(created_model, RFModel))

    def test_model_inputwarpedgp_arguments(self):
        '''
        Testing the arguments of the input warped GP
        '''
        model_type = 'input_warped_GP'
        exact_feval = True
        created_model = self.arguments_manager.model_creator(model_type, exact_feval, self.space)
        self.assertTrue(isinstance(created_model, InputWarpedGPModel))

    def test_model_warpedgo_arguments(self):
        '''
        Testing the arguments of the warped GP
        '''
        model_type = 'warpedGP'
        exact_feval = True
        created_model = self.arguments_manager.model_creator(model_type, exact_feval, self.space)
        self.assertTrue(isinstance(created_model, WarpedGPModel))

    def test_acquisition_ei_arguments(self):
        '''
        Testing the arguments of the Expected Improvement
        '''
        acquisition_type = 'EI'
        created_acquisition = self.arguments_manager.acquisition_creator(acquisition_type, self.model, self.space, self.acquisition_optimizer, self.cost)
        self.assertTrue(isinstance(created_acquisition, AcquisitionEI))
        self.assertEquals(created_acquisition.jitter,0.02)

    def test_acquisition_lcb_arguments(self):
        '''
        Testing the arguments of the Lower Confidence Bound
        '''
        acquisition_type = 'LCB'
        created_acquisition = self.arguments_manager.acquisition_creator(acquisition_type, self.model, self.space, self.acquisition_optimizer, self.cost)
        self.assertTrue(isinstance(created_acquisition, AcquisitionLCB))
        self.assertEquals(created_acquisition.exploration_weight,2.5)

    def test_evaluator_arguments(self):
        '''
        Testing the arguments of the local_penalization evaluator
        '''
        evaluator_type = 'local_penalization'
        batch_size = 2
        model_type = 'GP'
        created_evaluator = self.arguments_manager.evaluator_creator(evaluator_type, self.acquisition, batch_size, model_type, self.model, self.space, self.acquisition_optimizer)
        self.assertTrue(isinstance(created_evaluator, LocalPenalization))
        self.assertEquals(created_evaluator.acquisition.transform,'softplus')

# Copyright (c) 2019, author: Jose Hugo Elsas (jhelsas@tecgraf.puc-rio.br/jhelsas@gmail.com)
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import GPyOpt
import collections
import numpy as np
import time
import csv

from ..util.general import best_value, normalize
from ..util.duplicate_manager import DuplicateManager
from ..core.errors import InvalidConfigError
from ..core.task.cost import CostModel
from ..optimization.acquisition_optimizer import ContextManager
try:
    from ..plotting.plots_bo import plot_acquisition, plot_convergence
except:
    pass

from ..core.bo import BO

class ModularConstrainedBayesianOptimization(BO):
    """
    Modular Constrained Bayesian optimization. This class wraps the optimization loop around the different handlers.
        Addition to wrap the additional models for Probability of Feasibility constraint handling

    :param model: GPyOpt model class.
    :param space: GPyOpt space class.
    :param objective: GPyOpt objective class.
    :param acquisition: GPyOpt acquisition class.
    :param evaluator: GPyOpt evaluator class.
    :param X_init: 2d numpy array containing the initial inputs (one per row) of the model.
    :param Y_init: 2d numpy array containing the initial outputs (one per row) of the model.
    :param cost: GPyOpt cost class (default, none).
    :param normalize_Y: whether to normalize the outputs before performing any optimization (default, True).
    :param model_update_interval: interval of collected observations after which the model is updated (default, 1).
    :param de_duplication: instantiated de_duplication GPyOpt class.
    """
    
    def __init__(self, model, model_c, space, objective, constraint, acquisition, evaluator, 
                 X_init, Y_init=None, C_init=None, cost = None, normalize_Y = True, 
                 model_update_interval = 1, de_duplication=False):
        
        self.initial_iter = True
        self.modular_optimization = True
        
        # --- Create optimization space
        super(ModularConstrainedBayesianOptimization ,self).__init__(model                  = model,
                                                                     space                  = space,
                                                                     objective              = objective,
                                                                     acquisition            = acquisition,
                                                                     evaluator              = evaluator,
                                                                     X_init                 = X_init,
                                                                     Y_init                 = Y_init,
                                                                     cost                   = cost,
                                                                     normalize_Y            = normalize_Y,
                                                                     model_update_interval  = model_update_interval,
                                                                     de_duplication         = de_duplication)
        
        self.C = C_init
        self.C_init  = C_init
        self.model_c = model_c 
        self.constraint = constraint
        
    def run_optimization(self, max_iter = 0, max_time = np.inf,  eps = 1e-8, 
                         context = None, verbosity=False, save_models_parameters= True, 
                         report_file = None, evaluations_file = None, models_file=None):
        """
        Runs Bayesian Optimization for a number 'max_iter' of iterations (after the initial exploration data)

        :param max_iter: exploration horizon, or number of acquisitions. If nothing is provided optimizes the current acquisition.
        :param max_time: maximum exploration horizon in seconds.
        :param eps: minimum distance between two consecutive x's to keep running the model.
        :param context: fixes specified variables to a particular context (values) for the optimization run (default, None).
        :param verbosity: flag to print the optimization results after each iteration (default, False).
        :param report_file: file to which the results of the optimization are saved (default, None).
        :param evaluations_file: file to which the evalations are saved (default, None).
        :param models_file: file to which the model parameters are saved (default, None).
        """

        if self.objective is None:
            raise InvalidConfigError("Cannot run the optimization loop without the objective function")

        # --- Save the options to print and save the results
        self.verbosity = verbosity
        self.save_models_parameters = save_models_parameters
        self.report_file = report_file
        self.evaluations_file = evaluations_file
        self.models_file = models_file
        self.model_parameters_iterations = None
        self.context = context

        # --- Check if we can save the model parameters in each iteration
        if self.save_models_parameters == True:
            if not (isinstance(self.model, GPyOpt.models.GPModel) or isinstance(self.model, GPyOpt.models.GPModel_MCMC)):
                print('Models printout after each iteration is only available for GP and GP_MCMC models')
                self.save_models_parameters = False
                
            # TODO : modify this to a loop on all constrints 
            #if not (isinstance(self.model_c, GPyOpt.models.GPModel) or isinstance(self.model_c, GPyOpt.models.GPModel_MCMC)):
            #    print('Constrained Models printout after each iteration is only available for GP and GP_MCMC models')
            #    self.save_constrained_models_parameters = False

        # --- Setting up stop conditions
        self.eps = eps
        if  (max_iter is None) and (max_time is None):
            self.max_iter = 0
            self.max_time = np.inf
        elif (max_iter is None) and (max_time is not None):
            self.max_iter = np.inf
            self.max_time = max_time
        elif (max_iter is not None) and (max_time is None):
            self.max_iter = max_iter
            self.max_time = np.inf
        else:
            self.max_iter = max_iter
            self.max_time = max_time

        # --- TODO : update to track constraint cost
        # --- Initial function evaluation and model fitting
        if self.X is not None and self.Y is None:
            self.Y, cost_values = self.objective.evaluate(self.X)
            if self.cost.cost_type == 'evaluation_time':
                self.cost.update_cost_model(self.X, cost_values)

        # --- Initialize iterations and running time
        self.time_zero = time.time()
        self.cum_time  = 0
        self.num_acquisitions = 0
        self.suggested_sample = self.X
        self.Y_new = self.Y

        # --- Initialize time cost of the evaluations
        while (self.max_time > self.cum_time):
            # --- Update model
            try:
                self._update_model(self.normalization_type)
            except np.linalg.linalg.LinAlgError:
                break

            if (self.num_acquisitions >= self.max_iter
                    or (len(self.X) > 1 and self._distance_last_evaluations() <= self.eps)):
                break

            self.suggested_sample = self._compute_next_evaluations()

            # --- Augment X
            self.X = np.vstack((self.X,self.suggested_sample))

            # --- Evaluate *f* in X, augment Y and update cost function (if needed)
            self.evaluate_objective()
            
            # TODO - implement this in the loop
            # --- Evaluate *c* in C, augment C 
            self.evaluate_constraint()

            # --- Update current evaluation time and function evaluations
            self.cum_time = time.time() - self.time_zero
            self.num_acquisitions += 1

            if verbosity:
                print("num acquisition: {}, time elapsed: {:.2f}s".format(
                    self.num_acquisitions, self.cum_time))

        # TODO update this function
        # --- Stop messages and execution time
        self._compute_results()

        # --- Print the desired result in files
        if self.report_file is not None:
            self.save_report(self.report_file)
        if self.evaluations_file is not None:
            self.save_evaluations(self.evaluations_file)
        if self.models_file is not None:
            self.save_models(self.models_file)
                
    def evaluate_constraint(self):
        """
        Evaluates the objective
        """
        self.C_new, ccost_new = self.constraint.evaluate(self.suggested_sample)
        
        self.cost.update_cost_model(self.suggested_sample, ccost_new)
        self.C = np.vstack((self.C,self.C_new))
        
    def _update_model(self, normalization_type='stats'):
        """
        Updates the model (when more than one observation is available) and saves the parameters (if available).
        """
        if self.num_acquisitions % self.model_update_interval == 0:

            # input that goes into the model (is unziped in case there are categorical variables)
            X_inmodel = self.space.unzip_inputs(self.X)

            # Y_inmodel is the output that goes into the model
            if self.normalize_Y:
                Y_inmodel = normalize(self.Y, normalization_type)
            else:
                Y_inmodel = self.Y
                
            C_inmodel = self.C

            self.model.updateModel(X_inmodel, Y_inmodel, None, None)
            for i,mdl_c in enumerate(self.model_c):
                mdl_c.updateModel(X_inmodel,C_inmodel[:,i:(i+1)], None, None) # Updating the constraint model

        # Save parameters of the model
        self._save_model_parameter_values()
        
    def get_constraints_evaluations(self):
        return self.X.copy(), self.C.copy()
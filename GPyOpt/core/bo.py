# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
import time
from ..util.general import best_value, reshape
from ..core.task.cost import CostModel
try:
    from ..plotting.plots_bo import plot_acquisition, plot_convergence
except:
    pass

class BO(object):
    """
    Runner of Bayesian optimization loop. This class wraps the optimization loop around the different handlers.
    :param model: GPyOpt model class.
    :param space: GPyOpt space class.
    :param objective: GPyOpt objective class.
    :param acquisition: GPyOpt acquisition class.
    :param evaluator: GPyOpt evaluator class.
    :param X_init: 2d numpy array containing the initial inputs (one per row) of the model.
    :param Y_init: 2d numpy array containing the initial outputs (one per row) of the model.
    :param cost: GPyOpt cost class (defaut, none).
    :param normalize_Y: wheter to normalize the outputs before performing any optimization (default, True).
    :param model_update_interval: interval of collected observations after which the model is updated (default, 1).
    """


    def __init__(self, model, space, objective, acquisition, evaluator, X_init, Y_init=None, cost = None, normalize_Y = True, model_update_interval = 1):
        self.model = model
        self.space = space
        self.objective = objective
        self.acquisition = acquisition
        self.evaluator = evaluator
        self.normalize_Y = normalize_Y
        self.model_update_interval = model_update_interval
        self.X = X_init
        self.Y = Y_init
        self.cost = CostModel(cost)
        

    def run_optimization(self, max_iter = None, max_time = None,  eps = 1e-8, verbosity=True, report_file = None):
        """ 
        Runs Bayesian Optimization for a number 'max_iter' of iterations (after the initial exploration data)

        :param max_iter: exploration horizon, or number of acquisitions. If nothing is provided optimizes the current acquisition.  
        :param max_time: maximum exploration horizont in seconds.
        :param eps: minimum distance between two consecutive x's to keep running the model.
        :param vervosity: flag to print the optimization results after each iteration (default, True).
        :param report_file: filename of the file where the results of the optimization are saved (default, None).
        """
        self.verbosity = verbosity

        # --- Setting up stop conditions
        self.eps = eps 
        if  (max_iter == None) and (max_time == None):
            self.max_iter = 0
            self.max_time = np.inf
        elif (max_iter == None) and (max_time != None):
            self.max_iter = np.inf
            self.max_time = max_time
        elif (max_iter != None) and (max_time == None):
            self.max_iter = max_iter
            self.max_time = np.inf     
        else:
            self.max_iter = max_iter
            self.max_time = max_time     
            
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
                self._update_model()
            except np.linalg.linalg.LinAlgError:
                break

            # --- Update and optimize acquisition and compute the exploration level in the next evaluation
            self.suggested_sample = self._compute_next_evaluations()
            
            if not ((self.num_acquisitions < self.max_iter) and (self._distance_last_evaluations() > self.eps)): 
                break

            # --- Augment X
            self.X = np.vstack((self.X,self.suggested_sample))
            
            # --- Evaluate *f* in X, augment Y and update cost function (if needed)
            self.evaluate_objective()

            # --- Update current evaluation time and function evaluations
            self.cum_time = time.time() - self.time_zero  
            self.num_acquisitions += 1
                
   
        # --- Stop messages and execution time   
        self._compute_results()

        # --- Plot convergence results and print report
        self._print_convergence()
        if report_file != None:  self.save_report(report_file)


    def _print_convergence(self):
        """
        Prints the reason why the optimization stoped.
        """
        # --- Print stopping reason
        if self.verbosity: 
            if (self.num_acquisitions == self.max_iter) and (not self.initial_iter):        
                print('   ** Maximum number of iterations reached **')
                return 1
            elif (self._distance_last_evaluations() < self.eps) and (not self.initial_iter): 
                print('   ** Two equal location selected **')
                return 1
            elif (self.max_time < self.cum_time) and not (self.initial_iter):               
                print('   ** Evaluation time reached **')
                return 0

            if self.initial_iter:
                print('** GPyOpt Bayesian Optimization class initialized succesfuly **')
                self.initial_iter = False  



    def evaluate_objective(self):
        """
        Evaluates the objective
        """
        self.Y_new, cost_new = self.objective.evaluate(self.suggested_sample)
        self.cost.update_cost_model(self.suggested_sample, cost_new)
        self.Y = np.vstack((self.Y,self.Y_new))

    def _compute_results(self):
        """
        Computes the optimum and its value.
        """
        self.Y_best = best_value(self.Y)
        self.x_opt = self.X[np.argmin(self.Y),:]
        self.fx_opt = min(self.Y)


    def _distance_last_evaluations(self):
        """
        Computes the distance between the last two evaluations.
        """
        return np.sqrt(sum((self.X[self.X.shape[0]-1,:]-self.X[self.X.shape[0]-2,:])**2))  


    def _compute_next_evaluations(self):
        """
        Computes the location of the new evalaution (optimizes the acquisition in the standard case). 
        """
        return self.evaluator.compute_batch() 
        
    def _update_model(self):
        """
        Updates the model.
        """
        if (self.num_acquisitions%self.model_update_interval)==0:
            if self.normalize_Y:
                self.model.updateModel(self.X,(self.Y-self.Y.mean())/(self.Y.std()),self.suggested_sample,self.Y_new)
            else:
                self.model.updateModel(self.X, self.Y,self.suggested_sample,self.Y_new)


    def plot_acquisition(self,filename=None):
        """        
        Plots the model and the acquisition function.
            if self.input_dim = 1: Plots data, mean and variance in one plot and the acquisition function in another plot
            if self.input_dim = 2: as before but it separates the mean and variance of the model in two different plots
        :param filename: name of the file where the plot is saved
        """  
        return plot_acquisition(self.acquisition.space.get_bounds(),
                                self.model.model.X.shape[1],
                                self.model.model,
                                self.model.model.X,
                                self.model.model.Y,
                                self.acquisition.acquisition_function,
                                self.suggested_sample,
                                filename)


    def plot_convergence(self,filename=None):
        """
        Makes twp plots to evaluate the convergence of the model:
            plot 1: Iterations vs. distance between consecutive selected x's
            plot 2: Iterations vs. the mean of the current model in the selected sample.
        :param filename: name of the file where the plot is saved
        """
        return plot_convergence(self.X,self.Y_best,filename)
    
    def get_evaluations(self):
        return self.X.copy(), self.Y.copy()

    def save_report(self, report_file= 'GPyOpt-results.txt'):
        """
        Saves a report with the main resutls of the optimization.
 
        :param report_file: name of the file in which the results of the optimization are saved.
        """

        with open(report_file,'w') as file:
            file.write('---------------------------------' + ' Results file ' + '--------------------------------------\n')
            file.write('GPyOpt Version 1.0.0 \n')
            file.write('Date and time:              ' + time.strftime("%c")+'\n')
            if self.num_acquisitions==self.max_iter: 
                file.write('Optimization completed:     ' +'YES, ' + str(self.X.shape[0]).strip('[]') + ' samples collected.\n')
            else:
                file.write('Optimization completed:     ' +'NO,' + str(self.X.shape[0]).strip('[]') + ' samples collected.\n')
            file.write('Optimization time:          ' + str(self.time).strip('[]') +' seconds.\n')   
            
            file.write('---------------------------------' + ' Problem set up ' + '------------------------------------\n')   
            #file.write('Problem Name:          ' + str(self.input_dim).strip('[]') +'\n')            
            #file.write('Problem Dimension:          ' + str(self.input_dim).strip('[]') +'\n')    
            #file.write('Continous variables     ' + str(self.bounds).strip('[]') +'\n') 
            #file.write('Discrete variables     ' + str(self.bounds).strip('[]') +'\n') 
            #file.write('Bandits                ' + str(self.bounds).strip('[]') +'\n')            
            #file.write('Cost used:                  ' + str(self.bounds).strip('[]') +'\n') 

            file.write('-------------------------------' + ' Optimization set up ' + '----------------------------------\n')
 
            #file.write('Model type:                 ' + str(self.sparseGP).strip('[]') + '\n')  
            #file.write('Acquisition:                ' + self.acqu_name + '\n')   
            #file.write('Acquisition optimizer:      ' + self.acqu_optimize_method+ '\n') 
            #file.write('Batch method:               ' + self.acqu_name + '\n')             
            #file.write('Batch size:                 ' + str(self.n_inbatch).strip('[]') +'\n')  

            file.write('---------------------------------' + ' Summary ' + '------------------------------------------\n')
            file.write('Best found minimum:         ' + str(min(self.Y)).strip('[]') +'\n') 
            file.write('Minumum location:           ' + str(self.X[np.argmin(self.Y),:]).strip('[]') +'\n') 
    
            file.close()











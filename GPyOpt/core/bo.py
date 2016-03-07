# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
import time
from ..util.general import best_value, reshape
try:
    from ..plotting.plots_bo import plot_acquisition, plot_convergence
except:
    pass

class BO(object):
    def __init__(self, model, space, objective, acquisition_func, X_init, Y_init=None, cost = None, normalize_Y = True, model_update_interval = 1):
        self.model = model
        self.space = space
        self.objective = objective
        self.acquisition_func = acquisition_func
        self.cost = cost
        self.normalize_Y = normalize_Y
        self.model_update_interval = model_update_interval
        self.X_init = X_init
        self.Y_init = Y_init
        

    def run_optimization(self, max_iter = None, max_time = None,  eps = 1e-8, verbosity=True, report_file = None, **kargs):
        """ 
        Runs Bayesian Optimization for a number 'max_iter' of iterations (after the initial exploration data)

        :param max_iter: exploration horizon, or number of acquisitions. If nothing is provided optimizes the current acquisition.  
        :param max_time: maximum exploration horizont in seconds.
        :param eps: minimum distance between two consecutive x's to keep running the model
        """

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
        if self.X_init is not None and self.Y_init is None:
            self.X = self.X_init
            self.X_init = None
            self.Y, cost_values = self.objective.evaluate(self.X)
            self.cost.update_cost_model(self.X, cost_values)
        
        elif self.cost.cost_type == 'evaluation_time':
            self.Y, cost_values = self.objective.evaluate(self.X)
            self.cost.update_cost_model(self.X, cost_values)
        
        # --- Initialize iterations and running time
        self.time_zero = time.time()
        self.cum_time  = 0
        self.num_acquisitions = 0

        # --- Initialize time cost of the evaluations
        while (self.max_time > self.cum_time):
            # --- Update model
            try:
                self._update_model()
            except np.linalg.linalg.LinAlgError:
                break

            # --- Update and optimize acquisition and compute the exploration level in the next evaluation
            self.suggested_sample = self._optimize_acquisition()
            self._compute_exploration_next_evaluation()
            
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
        self._print_convergence(verbosity)
        if report_file != None:  self.save_report(report_file)


    def _print_convergence(self,verbose):
        # --- Print stopping reason
        if verbose: 
            print '*Optimization completed:'
        if (self.num_acquisitions > self.max_iter):        
            print '   -Maximum number of iterations reached.' 
            return 1
        if (self._distance_last_evaluations() < self.eps): 
            print '   -Method converged.'
            return 1
        if (self.max_time < self.cum_time):               
            print '   -Evaluation time reached.'
            return 0

    def evaluate_objective(self):
        Y_new, cost_new = self.objective.evaluate(self.suggested_sample)
        self.cost.update_cost_model(self.suggested_sample, cost_new)
        self.Y = np.vstack((self.Y,Y_new))


    def _compute_exploration_next_evaluation(self):           
        if self.num_acquisitions == 0:
            self.exploration_in_samples = self.model.predict(self.X)[1]
        else:
            self.exploration_in_samples = np.vstack((self.exploration_in_samples,
                                                    self.model.predict(self.suggested_sample)[1]))

    def _compute_results(self):
        self.Y_best = best_value(self.Y)
        self.x_opt = self.X[np.argmin(self.Y),:]
        self.fx_opt = min(self.Y)


    def _distance_last_evaluations(self):
        return np.sqrt(sum((self.X[self.X.shape[0]-1,:]-self.X[self.X.shape[0]-2,:])**2))  


    def _optimize_acquisition(self):
        return self.acquisition_func.optimize()

        
    def _update_model(self):
        if (self.num_acquisitions%self.model_update_interval)==0:
            if self.normalize_Y:
                self.model.updateModel(self.X,(self.Y-self.Y.mean())/(self.Y.std()),None,None)
            else:
                self.model.updateModel(self.X, self.Y,None,None)


    def plot_acquisition(self,filename=None):
        """        
        Plots the model and the acquisition function.
            if self.input_dim = 1: Plots data, mean and variance in one plot and the acquisition function in another plot
            if self.input_dim = 2: as before but it separates the mean and variance of the model in two different plots
        :param filename: name of the file where the plot is saved
        """  
        return plot_acquisition(self.acquisition_func.space.get_bounds(),
                                self.model.model.X.shape[1],
                                self.model.model,
                                self.model.model.X,
                                self.model.model.Y,
                                self.acquisition_func.acquisition_function,
                                self.suggested_sample,
                                filename)


    def plot_convergence(self,filename=None):
        """
        Makes three plots to evaluate the convergence of the model
            plot 1: Iterations vs. distance between consecutive selected x's
            plot 2: Iterations vs. the mean of the current model in the selected sample.
            plot 3: Iterations vs. the variance of the current model in the selected sample.
        :param filename: name of the file where the plot is saved
        """
        return plot_convergence(self.X,self.Y_best,self.exploration_in_samples,filename)
    
    def get_evaluations(self):
        return self.X.copy(), self.Y.copy()

    def save_report(self, report_file= 'GPyOpt-results.txt'):

        ##
        ## TODO
        ##
        """
        Save a report with the results of the optimization. A file is produced every 
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











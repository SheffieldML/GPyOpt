# Copyright (c) 2015, Javier Gonzalez
# Copyright (c) 2015, the GPy Authors (see GPy AUTHORS.txt)
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import GPy
import deepgp
import numpy as np
import time
from ..util.general import best_value, reshape, spawn, evaluate_function
from ..core.optimization import lp_batch_optimization, random_batch_optimization, predictive_batch_optimization
try:
    from ..plotting.plots_bo import plot_acquisition, plot_convergence
except:
    pass

class BO(object):
    def __init__(self, func, model, space, acquisition_func, acq_optimizer, normalize_Y=True, model_optimize_interval=1):
        self.f = func
        self.model = model
        self.space = space
        self.acquisition_func = acquisition_func
        self.acq_optimizer = acq_optimizer 
        self.normalize_Y = normalize_Y
        self.model_optimize_interval = model_optimize_interval
        
    def run_optimization(self, max_iter = None, max_time = None,  eps = 1e-8, verbose=True):
        """ 
        Runs Bayesian Optimization for a number 'max_iter' of iterations (after the initial exploration data)

        :param max_iter: exploration horizon, or number of acquisitions. It nothing is provided optimizes the current acquisition.  
        :param max_time: maximum exploration horizont in seconds.

	    :param n_inbatch: number of samples to collected everytime *f* is evaluated (one by default).
        :param acqu_optimize_method: method to optimize the acquisition function 
            -'DIRECT': uses the DIRECT algorithm of Jones and Stuckmann. 
            -'CMA': uses the Covariance Matrix Adaptation Algorithm.
	        -'brute': Run local optimizers in a grid of points.
	        -'random': Run local optimizers started at random locations.
            -'fast_brute': the same as brute but runs only one optimizer in the best location. It is used by default.
            -'fast_random': the same as random but runs only one optimizer in the best location.
        :param acqu_optimize_restarts: numbers of random restarts in the optimization of the acquisition function, default = 20.
	    :param batch_method: method to collect samples in batches
            -'predictive': uses the predicted mean in the selected sample to update the acquisition function.
            -'lp': used a penalization of the acquisition function to based on exclusion zones.
            -'random': collects the element of the batch randomly
    	:param eps: minimum distance between two consecutive x's to keep running the model
    	:param n_procs: The number of processes used for evaluating the given function *f* (ideally nprocs=n_inbatch).
        :param true_gradients: If the true gradients (can be slow) of the acquisition ar an approximation is used (True, default).
        :param save_interval: number of iterations after which a file is produced with the current results.
    
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

            # --- Update and optimize acquisition
            self.suggested_sample = self._optimize_acquisition()

            # --- Update internal elements (needed for plotting)
            self._update_internal_elements()
            
            if not ((self.num_acquisitions < self.max_iter) and (self._distance_last_evaluations() > self.eps)): 
                break

            # --- Augment X
            self.X = np.vstack((self.X,self.suggested_sample))
            
            # --- Evaluate *f* in X and augment Y
            self.evaluate_objective()

            # --- Update current evaluation time and function evaluations
            self.cum_time = time.time() - self.time_zero  
            self.num_acquisitions += 1
                
   
        # --- Stop messages and execution time   
        self._compute_results()

        # --- Plot convergence results
        self._print_convergence(verbose)



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
        if self.n_procs==1:
            Y_new, Y_costnew = evaluate_function(self.f,self.suggested_sample)
            self.Y = np.vstack((self.Y,Y_new))
            self.Y_cost= np.vstack((self.Y_cost,Y_costnew))
        else:
            try:
                # --- Parallel evaluation of *f* if several cores are available
                from multiprocessing import Process, Pipe
                from itertools import izip          
                divided_samples = [self.suggested_sample[i::self.n_procs] for i in xrange(self.n_procs)]
                pipe=[Pipe() for i in xrange(self.n_procs)]
                proc=[Process(target=spawn(self.f),args=(c,x)) for x,(p,c) in izip(divided_samples,pipe)]
                [p.start() for p in proc]
                [p.join() for p in proc]
                rs = [p.recv() for (p,c) in pipe]
                self.Y = np.vstack([self.Y]+rs)
            except:
                if not hasattr(self, 'parallel_error'):
                    print 'Error in parallel computation. Fall back to single process!'
                    self.parallel_error = True 
                self.Y = np.vstack((self.Y,self.f(np.array(self.suggested_sample))))


    def _update_internal_elements(self):           
        if self.num_acquisitions == 0:
            pred_min = self.model.predict(self.X)
            self.s_in_min = np.sqrt(abs(pred_min[1]))
        else:
            pred_min = self.model.predict(reshape(self.suggested_sample,self.input_dim))
            self.s_in_min = np.vstack((self.s_in_min,np.sqrt(abs(pred_min[1])))) 

    def _compute_results(self):
        self.Y_best = best_value(self.Y)
        self.x_opt = self.X[np.argmin(self.Y),:]
        self.fx_opt = min(self.Y)

    def _distance_last_evaluations(self):
        return np.sqrt(sum((self.X[self.X.shape[0]-1,:]-self.X[self.X.shape[0]-2,:])**2))  

    def _optimize_acquisition(self):
        """
        Optimizes the acquisition function. This function selects the type of batch method and passes the arguments for the rest of the optimization.

        """
        return self.acquisition_func.optimize()
        

    def _update_model(self):
        """        
        Updates X and Y in the model and re-optimizes the parameters of the new model

        """
        if (self.num_acquisitions%self.model_optimize_interval)==0:
            if self.normalize_Y:
                self.model.updateModel(self.X,(self.Y-self.Y.mean())/(self.Y.std()), None, None)
            else:
                self.model.updateModel(self.X, self.Y, None, None)

    def plot_acquisition(self,filename=None):
        """        
        Plots the model and the acquisition function.
            if self.input_dim = 1: Plots data, mean and variance in one plot and the acquisition function in another plot
            if self.input_dim = 2: as before but it separates the mean and variance of the model in two different plots
        :param filename: name of the file where the plot is saved
        """  
        return plot_acquisition(self.bounds,self.input_dim,self.model,self.model.X,self.model.Y,self.acquisition_func.acquisition_function,self.suggested_sample,filename)

    def plot_convergence(self,filename=None):
        """
        Makes three plots to evaluate the convergence of the model
            plot 1: Iterations vs. distance between consecutive selected x's
            plot 2: Iterations vs. the mean of the current model in the selected sample.
            plot 3: Iterations vs. the variance of the current model in the selected sample.
        :param filename: name of the file where the plot is saved
        """
        return plot_convergence(self.X,self.Y_best,self.s_in_min,filename)
    
    def get_evaluations(self):
        return self.X.copy(), self.Y.copy()

    def save_report(self, report_file= 'GPyOpt-results.txt ' ):
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
            file.write('Problem Dimension:          ' + str(self.input_dim).strip('[]') +'\n')    
            file.write('Problem bounds:             ' + str(self.bounds).strip('[]') +'\n') 
            file.write('Batch size:                 ' + str(self.n_inbatch).strip('[]') +'\n')    
            file.write('Acquisition:                ' + self.acqu_name + '\n')  
            file.write('Acquisition optimizer:      ' + self.acqu_optimize_method+ '\n')  
            file.write('Sparse GP:                  ' + str(self.sparseGP).strip('[]') + '\n')  
            file.write('---------------------------------' + ' Summary ' + '------------------------------------------\n')
            file.write('Best found minimum:         ' + str(min(self.Y)).strip('[]') +'\n') 
            file.write('Minumum location:           ' + str(self.X[np.argmin(self.Y),:]).strip('[]') +'\n') 
    
            file.close()











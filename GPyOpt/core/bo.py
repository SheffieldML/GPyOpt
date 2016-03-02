# Copyright (c) 2015, Javier Gonzalez
# Copyright (c) 2015, the GPy Authors (see GPy AUTHORS.txt)
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from __future__ import print_function

import numpy as np
import time

from ..util.general import best_value
from ..util.general import reshape
from ..util.general import spawn
from ..core.optimization import lp_batch_optimization
from ..core.optimization import random_batch_optimization
from ..core.optimization import predictive_batch_optimization

try:
    from ..plotting.plots_bo import plot_acquisition
    from ..plotting.plots_bo import plot_convergence
except:
    pass


class BO(object):
    def __init__(self, acquisition_func):
        self.acquisition_func = acquisition_func

    def _init_model(self):
        pass

    def run_optimization(self, max_iter = None, n_inbatch=1, acqu_optimize_method='fast_random', acqu_optimize_restarts=200, batch_method='predictive',
        eps = 1e-8, n_procs=1, true_gradients = True, verbose=True):
        """
        Runs Bayesian Optimization for a number 'max_iter' of iterations (after the initial exploration data)

        :param max_iter: exploration horizon, or number of acquisitions. It nothing is provided optimizes the current acquisition.
	    :param n_inbatch: number of samples to collected everytime *f* is evaluated (one by default).
        :param acqu_optimize_method: method to optimize the acquisition function
            -'grid': optimizes the acquisition in fixed grid of points.
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
        # --- Load the parameters of the function into the object.
        if max_iter == None:
            self.max_iter = 10*self.input_dim
        else:
            self.max_iter = max_iter

        self.num_acquisitions = 0
        self.n_inbatch=n_inbatch
        self.batch_method = batch_method
        if batch_method=='lp':
            from .acquisition import AcquisitionMP
            if not isinstance(self.acquisition_func, AcquisitionMP):
                self.acquisition_func = AcquisitionMP(self.acquisition_func, self.acquisition_func.acquisition_par)
        self.eps = eps
        self.acqu_optimize_method = acqu_optimize_method
        self.acqu_optimize_restarts = acqu_optimize_restarts
        self.acquisition_func.set_model(self.model)
        self.n_procs = n_procs

        # --- Decide wether we use the true gradients to optimize the acquitision function
        if true_gradients !=True:
            self.true_gradients = False
            self.acquisition_func.d_acquisition_function = None
        else:
            self.true_gradients = true_gradients

        # --- Get starting of running time
        self.time = time.time()

        # --- If this is the first time to optimization is run - update the model and normalize id needed
        if self.first_time_optimization:
            self._update_model()
            prediction = self.model.predict(self.X)
            self.s_in_min = np.sqrt(abs(prediction[1]))
            self.first_time_optimization = False

        # --- Initialization of stop conditions.
        k=0
        distance_lastX = np.sqrt(sum((self.X[self.X.shape[0]-1,:]-self.X[self.X.shape[0]-2,:])**2))

        # --- BO loop: this loop does the hard work.
        while k<self.max_iter and distance_lastX > self.eps:

            # --- Augment X
            self.X = np.vstack((self.X,self.suggested_sample))

            # --- Evaluate *f* in X and augment Y
            if self.n_procs==1:
                self.Y = np.vstack((self.Y,self.f(np.array(self.suggested_sample))))
            else:
                try:
                    # --- Parallel evaluation of *f* if several cores are available
                    from multiprocessing import Process, Pipe
                    from itertools import izip
                    divided_samples = [self.suggested_sample[i::self.n_procs] for i in range(self.n_procs)]
                    pipe=[Pipe() for i in range(self.n_procs)]
                    proc=[Process(target=spawn(self.f),args=(c,x)) for x,(p,c) in izip(divided_samples,pipe)]
                    [p.start() for p in proc]
                    [p.join() for p in proc]
                    rs = [p.recv() for (p,c) in pipe]
                    self.Y = np.vstack([self.Y]+rs)
                except:
                    if not hasattr(self, 'parallel_error'):
                        print('Error in parallel computation. Fall back to single process!')
                        self.parallel_error = True
                    self.Y = np.vstack((self.Y,self.f(np.array(self.suggested_sample))))

            # --- Update internal elements (needed for plotting)
            self.num_acquisitions += 1
            pred_min = self.model.predict(reshape(self.suggested_sample,self.input_dim))
            self.s_in_min = np.vstack((self.s_in_min,np.sqrt(abs(pred_min[1]))))

            # --- Update model
            try:
                self._update_model()
            except np.linalg.linalg.LinAlgError:
                break

            # --- Update stop conditions
            k +=1
            distance_lastX = np.sqrt(sum((self.X[self.X.shape[0]-1,:]-self.X[self.X.shape[0]-2,:])**2))

        # --- Stop messages and execution time
        self.Y_best = best_value(self.Y)
        self.x_opt = self.X[np.argmin(self.Y),:]
        self.fx_opt = min(self.Y)
        self.time   = time.time() - self.time

        # --- Print stopping reason
        if verbose: print('*Optimization completed:')
        if k==self.max_iter and distance_lastX > self.eps:
            if verbose: print('   -Maximum number of iterations reached.')
            return 1
        else:
            if verbose: print('   -Method converged.')
            return 0


    def change_to_sparseGP(self, num_inducing):
        """
        Changes standard GP estimation to sparse GP estimation

	    :param num_inducing: number of inducing points for sparse-GP modeling
	     """
        if self.sparse == True:
            raise Exception('Sparse GP is already in use')
        else:
            self.num_inducing = num_inducing
            self.sparse = True
            self._init_model(self.X,self.Y)

    def change_to_standardGP(self):
        """
        Changes sparse GP estimation to standard GP estimation

        """
        if self.sparse == False:
            raise Exception('Sparse GP is already in use')
        else:
            self.sparse = False
            self._init_model(self.X,self.Y)


    def _optimize_acquisition(self):
        """
        Optimizes the acquisition function. This function selects the type of batch method and passes the arguments for the rest of the optimization.

        """
        # ------ Elements of the acquisition function
        acqu_name = self.acqu_name
        acquisition = self.acquisition_func.acquisition_function
        d_acquisition = self.acquisition_func.d_acquisition_function
        acquisition_par = self.acquisition_par
        model = self.model

        # ------  Parameters to optimize the acquisition
        acqu_optimize_restarts = self.acqu_optimize_restarts
        acqu_optimize_method = self.acqu_optimize_method
        n_inbatch = self.n_inbatch
        bounds = self.bounds

        # ------ Selection of the batch method (if any, predictive used when n_inbathc=1)
        if self.batch_method == 'predictive':
            X_batch = predictive_batch_optimization(acqu_name, acquisition_par, acquisition, d_acquisition, bounds, acqu_optimize_restarts, acqu_optimize_method, model, n_inbatch)
        elif self.batch_method == 'lp':
            X_batch = lp_batch_optimization(self.acquisition_func, bounds, acqu_optimize_restarts, acqu_optimize_method, model, n_inbatch)
        elif self.batch_method == 'random':
            X_batch = random_batch_optimization(acquisition, d_acquisition, bounds, acqu_optimize_restarts,acqu_optimize_method, model, n_inbatch)
        return X_batch


    def _update_model(self):
        """
        Updates X and Y in the model and re-optimizes the parameters of the new model

        """
        # ------- Normalize acquisition function (if needed)
        if self.normalize:
            self.model.set_XY(self.X,(self.Y-self.Y.mean())/(self.Y.std()))
        else:
            self.model.set_XY(self.X,self.Y)

        # ------- Optimize model when required
        if (self.num_acquisitions%self.model_optimize_interval)==0:
            self.model.optimization_runs = [] # clear previous optimization runs so they don't get used.
            self.model.optimize_restarts(num_restarts=self.model_optimize_restarts, verbose=self.verbosity)

        # ------- Optimize acquisition function
        self.suggested_sample = self._optimize_acquisition()


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

    def save_report(self, report_file= 'GPyOpt-results.txt' ):
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

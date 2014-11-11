import GPy
import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
import scipy
import random

from ..util.general import samples_multidimensional_uniform, multigrid, reshape, ellipse 
from ..core.optimization import grid_optimization, multi_init_optimization
from ..plotting.plots_bo import plot_acquisition, plot_convergence


#from .acquisition import AcquisitionEI 

class BO(object):
    def __init__(self, acquisition_func, bounds=None, model_optimize_interval=1, model_optimize_restarts=5, Nrandom=None, verbosity=1):
        if bounds==None: 
            print 'Box contrainst are needed. Please insert box constrains'    
        else:
            self.bounds = bounds
            self.input_dim = len(self.bounds)        
        self.acquisition_func = acquisition_func
        self.model_optimize_interval, self.model_optimize_restarts = model_optimize_interval, model_optimize_restarts
        self.Ngrid = 100
        if Nrandom ==None: self.Nrandom = 2*self.input_dim # number or samples for initial random exploration
        else: self.Nrandom = Nrandom  
        self.verbosity=verbosity
    
 
    def _init_model(self, X, Y):
        pass
        
    def start_optimization(self, f, H, X=None, Y=None, Ninit=None):
        """ 
        Starts Bayesian Optimization for a number H of iterations (after the initial exploration data)

        :param: f the function to optimize
        :param X: input observations
        :param Y: output values
        :param H: exploration horizon, or number of iterations  

        ..Note : X and Y can be None. In this case Nrandom*model_dimension data are uniformly generated to initialize the model.
    
        """
        self.num_acquisitions = 0
        if f==None: print 'Function to optimize is requiered'
        else: self.f = f
        if H == None: H=0
        if X==None or Y == None:
            self.X = samples_multidimensional_uniform(self.bounds, self.Nrandom)
            self.Y = f(self.X)
        else:
            self.X = X
            self.Y = Y
        if Ninit == None: 
            self.Ninit = 10
        else: 
            self.Ninit = Ninit
        self._init_model(self.X, self.Y)
        self.acquisition_func.model = self.model
        self._update_model()
        prediction = self.model.predict(self.X)
        self.m_in_min = prediction[0]
        self.s_in_min = np.sqrt(prediction[1]) 
        self.optimization_started = True
        return self.continue_optimization(H)
    
    def change_to_sparseGP(self, num_inducing):
        """
        Changes standard GP estimation to sparse GP estimation
        """
        if self.sparse == True:
            raise 'Sparse GP is already in use'
        else:
            self.num_inducing = num_inducing
            self.sparse = True
            self._init_model(self.X,self.Y)

    def change_to_standardGP(self):
        """
        Changes sparse GP estimation to standard GP estimation
        """
        if self.sparse == False:
            raise 'Sparse GP is already in use'
        else:
            self.num_inducing = num_inducing
            self.sparse = False
            self._init_model(self.X,self.Y)

    def continue_optimization(self,H):
        """
        Continues Bayesian Optimization for a number H of iterations. Requieres prior initialization with self.start_optimization
        :param H: new exploration horizon, or number of extra iterations  

        """
        if self.optimization_started:
            k=1
            while k<=H:
                self.X = np.vstack((self.X,self.suggested_sample))
                self.Y = np.vstack((self.Y,self.f(np.array([self.suggested_sample]))))
                self.num_acquisitions += 1
                pred_min = self.model.predict(reshape(self.suggested_sample,self.input_dim))
                self.m_in_min = np.vstack((self.m_in_min,pred_min[0]))
                self.s_in_min = np.vstack((self.s_in_min,np.sqrt(pred_min[1])))
                try:
                    self._update_model()                
                except np.linalg.linalg.LinAlgError:
                    # Kernel become singular, DO something
                    print 'Optimization stopped. Two equal points selected.'
                    break
                k +=1
            return self.suggested_sample

        else: print 'Optimization not initiated: Use .start_optimization and provide a function to optimize'
        
    def _optimize_acquisition(self):
        """
        Optimizes the acquisition function. It combines initial grid search with local optimzation starting on the minimum of the grid

        """
        return multi_init_optimization(self.acquisition_func.acquisition_function,self.bounds, self.Ninit)
        #return density_sampling_optimization(self.acquisition_function, self.bounds, self.model)
        # return grid_optimization(self.acquisition_func.acquisition_function, self.bounds, self.Ngrid) 


    def _update_model(self):
        """        
        Updates X and Y in the model and re-optimizes the parameters of the new model

        """        
        self.model.set_XY(self.X,(self.Y-self.Y.mean())/self.Y.std())
        if (self.num_acquisitions%self.model_optimize_interval)==0:
            self.model.optimization_runs = [] # clear previous optimization runs so they don't get used.
            self.model.optimize_restarts(num_restarts=self.model_optimize_restarts, verbose=self.verbosity)            
        self.suggested_sample = self._optimize_acquisition()

    def plot_acquisition(self):
        """        
        Plots the model and the acquisition function.
            if self.input_dim = 1: Plots data, mean and variance in one plot and the acquisition function in another plot
            if self.input_dim = 2: as before but it separates the mean and variance of the model in two different plots

        """  
        return plot_acquisition(self.bounds,self.input_dim,self.model,self.X,self.Y,self.acquisition_func.acquisition_function,self.suggested_sample)

    def plot_convergence(self):
        """
        Makes three plots to evaluate the convergence of the model
            plot 1: Iterations vs. distance between consecutive selected x's
            plot 2: Iterations vs. the mean of the current model in the selected sample.
            plot 3: Iterations vs. the variance of the current model in the selected sample.

        """
        return plot_convergence(self.X,self.m_in_min,self.s_in_min)
























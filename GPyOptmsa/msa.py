import numpy as np
import GPy
from GPyOpt.core.optimization import wrapper_lbfgsb, wrapper_DIRECT, estimate_L, estimate_Min
from GPyOpt.core.acquisition import AcquisitionEL
from .util.general import samples_multidimensional_uniform, reshape, best_value, multigrid
from .util.acquisition import loss_nsahead
import GPyOptmsa
from .plotting.plots_bo import plot_acquisition, plot_convergence


class GLASSES:
    '''
    Class to run Bayesian Optimization with multiple steps ahead
    '''

    def __init__(self,f,bounds,X,Y,n_ahead = 1,exact_feval=False):
        self.input_dim = len(bounds)
        self.bounds = bounds
        self.X = X
        self.Y = Y
        
        # Initialize model
        self.kernel = GPy.kern.RBF(self.input_dim, variance=.1, lengthscale=.1)  + GPy.kern.Bias(self.input_dim)
        self.model  = GPy.models.GPRegression(X,Y,kernel= self.kernel)
        
        if exact_feval==True:
            self.model.Gaussian_noise.constrain_fixed(0.0001)
        else:
            self.model.Gaussian_noise.constrain_bounded(1e-2,1e6) #to avoid numerical problems

        self.model.optimize_restarts(5)
        
        # Imitilize loss
        self.loss = AcquisitionEL()
        self.update_loss()
        self.s_in_min = np.sqrt(self.model.predict(self.X)[1])
        self.f = f
        self.n_ahead = n_ahead # deault steps ahead. To acess this when none optimzation has been done yet
        self.L = estimate_L(self.model,self.bounds)
        self.Min = estimate_Min(self.model,self.bounds)


    def loss_ahead(self,x):
        return loss_nsahead(x, self.bounds, self.loss, self.n_ahead, self.L, self.Min, self.model)

    def run_optimization(self,max_iter, eps= 10e-6, ahead_remaining = False):

        # weigth of the previous acquisition in the dpp sample
        #self.n_samples_dpp = n_samples_dpp
        # Check the number of steps ahead to look at
        #if n_ahead==None:
        #    self.n_ahead = max_iter
        #else:
        #    self.n_ahead = n_ahead

        # initial stop conditions
        k = 1
        distance_lastX = np.sqrt(sum((self.X[self.X.shape[0]-1,:]-self.X[self.X.shape[0]-2,:])**2))

        while k<=max_iter and distance_lastX > eps:

            if ahead_remaining == True:
                self.n_ahead = max_iter-k+1

            # update loss
            self.update_loss()

            # Optimize loss
            X_new = wrapper_DIRECT(self.loss_ahead,self.bounds)

            self.suggested_sample = X_new

            # Augment the dataset
            self.X = np.vstack((self.X,X_new))
            self.Y = np.vstack((self.Y,self.f(X_new)))

            # Update the model
            self.model.set_XY(self.X,self.Y)
            self.model.optimize_restarts(verbose=False)

            # Update steps ahead if needed
            self.s_in_min = np.vstack((self.s_in_min,np.sqrt(abs(self.model.predict(X_new)[1]))))
            print 'Iteration ' + str(k) +' completed'
            k += 1

        self.Y_best = best_value(self.Y)


    def update_loss(self):
        self.loss.set_model(self.model) 

    def plot_loss(self,filename=None):
        """        
        Plots the model and the acquisition function.
            if self.input_dim = 1: Plots data, mean and variance in one plot and the acquisition function in another plot
            if self.input_dim = 2: as before but it separates the mean and variance of the model in two different plots
        :param filename: name of the file where the plot is saved
        """  
        return plot_acquisition(self.bounds,self.input_dim,self.model,self.model.X,self.model.Y,self.loss,self.suggested_sample,filename)

    def plot_convergence(self,filename=None):
        """
        Makes three plots to evaluate the convergence of the model
            plot 1: Iterations vs. distance between consecutive selected x's
            plot 2: Iterations vs. the mean of the current model in the selected sample.
            plot 3: Iterations vs. the variance of the current model in the selected sample.
        :param filename: name of the file where the plot is saved
        """
        return plot_convergence(self.X,self.Y_best,self.s_in_min,filename)





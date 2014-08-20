'''
Examples of use of the class BayesianOptimization
	- branin function

''' 
import numpy as np
from scipy.optimize import minimize
import GPyOpt

#
# Example 1: Optimization of the branin function
#

# create the object function
f_true = GPyOpt.fmodels.experiments2d.branin()
f_sim = GPyOpt.fmodels.experiments2d.branin(sd= .5)
f_true.plot()
bounds = f_true.bounds
H = 3

# starts the optimization with 3 data points 
myBopt = GPyOpt.methods.BayesianOptimization.BayesianOptimization(bounds, acquisition_type='MPI', acquisition_par = 0.01)
myBopt.start_optimization(f_sim.f,H=H)
myBopt.plot_acquisition()

# cotinue optimization for 10 observations more
myBopt.continue_optimization(H=50)
myBopt.plot_acquisition()
myBopt.plot_convergence()
myBopt.suggested_sample
f_true.min
f_true.fmin












'''
Examples of use of the class BayesianOptimization
	- forrester (1d)
	- branin (2d))
	- gSobol (arbitrary dimension)

''' 
import numpy as np
from scipy.optimize import minimize
import GPyOpt

#
# Example 1: Optimization of the forrester function
#

# create the object function
f_true = GPyOpt.fmodels.experiments1d.forrester()
f_sim = GPyOpt.fmodels.experiments1d.forrester(sd= .5)
#f_true.plot()
bounds = [(0,1)]
H = 0

# starts the optimization with 3 data points 
myBopt = GPyOpt.methods.BayesianOptimization.BayesianOptimization(bounds, acquisition_type='MPI', acquisition_par = 0.01)
myBopt.start_optimization(f_sim.f,H=H)
myBopt.plot_acquisition()

# cotinue optimization for 10 observations more
myBopt.continue_optimization(H=25)
myBopt.model.plot()
myBopt.plot_acquisition()
myBopt.plot_convergence()













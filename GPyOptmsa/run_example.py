import GPyOptmsa
import GPyOpt
import GPy
from numpy.random import seed
import numpy as np
from GPyOptmsa.util.general import samples_multidimensional_uniform, multigrid
from GPyOpt.fmodels.experiments2d import *

import warnings
warnings.filterwarnings("ignore")
seed(12345)


## Setup    
func            = cosines(sd=.1)
n_init          = 5               # number of initial points (per dimension).
max_iter        = 20              # Number of iterations (per dimension).

f         = func.f 
bounds    = func.bounds
input_dim = len(bounds)
    
# --- Matrices to save results
res_GLASSES_H    = np.empty([1,2])
res_EL           = np.empty([1,2])

# --- inital points
X = samples_multidimensional_uniform(bounds,n_init*len(bounds))
Y = f(X)

# --- Full GLASSES: considers as number of steps ahead the remaining number of evaluations
GLASSES_H    = GPyOptmsa.msa.GLASSES(f,bounds, X,Y)
GLASSES_H.run_optimization(max_iter=max_iter,ahead_remaining = True) 
GLASSES.plot_convergence() 

# --- Expected loss
EL           = GPyOpt.methods.BayesianOptimization(f=f,bounds=bounds, X=X, Y=Y ,acquisition='EL') 
EL.run_optimization(max_iter=max_iter, acqu_optimize_method='DIRECT')  
EL.plot_convergence()





    
















    
    
    
    








    
    
    
    

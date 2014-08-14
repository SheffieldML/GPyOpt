'''
Example of use of the classes

- experiments.py: simulates data from experiments using different classes of functions
		- branin (2d)
		- forrester (1d)
		- gSobol (arbitrary dimension)

'''
import numpy as np
from scipy.optimize import minimize

run experiments.py
run BayesianOptimization.py

## Example branin function
branin_exp = branin(sd=10)
x = np.array([1.3, 0.7])
branin_exp.f(x)
branin_exp.plot()
res = minimize(branin_exp.f, x0=np.array([1,1]), method='nelder-mead',options={'xtol': 1e-8, 'disp': True})

# Example forrester fucntion
forr_exp = forrester()
x = np.array([2.2,4,4,4])
forr_exp.f(x)
forr_exp.plot()

# Example for the gSobol function
a = np.array([1,.5,.1]) 
x = np.array([[1,1,1],[2,3,4],[33,4,5],[43,2,2]])
sobol_exp = gSobol(a=a)
sobol_exp.f(x)
sobol_exp.S_coef














import numpy as np
from ..util.general import multigrid, samples_multidimensional_uniform
import scipy

def grid_optimization(acquisition_function, bounds, Ngrid):
	grid = multigrid(bounds,Ngrid)
	pred_grid = acquisition_function(grid)
	x0 =  grid[np.argmin(pred_grid)]
	res = scipy.optimize.minimize(acquisition_function, x0=np.array(x0),method='SLSQP',bounds=bounds)
	return res.x

def multi_init_optimization(acquisition_function, bounds, Ninit):
	# sample Ninit initial points 
	mins = np.zeros((Ninit,len(bounds)))
	fmins = np.zeros(Ninit)
	samples = samples_multidimensional_uniform(bounds,Ninit)
	for k in range(Ninit):
		res = scipy.optimize.minimize(acquisition_function, x0 = samples[k,:] ,method='SLSQP',bounds=bounds)
		mins[k] = res.x
		fmins[k] = res.fun
	return mins[np.argmin(fmins)]


#def density_sampling_optimization(acquisition_function, bounds, model, X):




	

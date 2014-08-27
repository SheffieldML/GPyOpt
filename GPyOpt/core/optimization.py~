import numpy as np
from ..util.general import multigrid
import scipy

def grid_optimization(acquisition_function, bounds, Ngrid):
	grid = multigrid(bounds,Ngrid)
	pred_grid = acquisition_function(grid)
	x0 =  grid[np.argmin(pred_grid)]
	res = scipy.optimize.minimize(acquisition_function, x0=np.array(x0),method='SLSQP',bounds=bounds)
	return res.x

#def multi_init_optimization(acquisition_function, bounds, Ninit):

#def density_sampling_optimization(acquisition_function, bounds, model, X):




	

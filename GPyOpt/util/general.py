import numpy as np
from scitools.BoxGrid import UniformBoxGrid



## generates a multidimensional grid uniformly distributes
def samples_multimensional_uniform(bounds,num_data):
        dim = len(bounds)
        Z_rand = np.zeros(shape=(num_data,dim))
        for k in range(0,dim): Z_rand[:,k] = np.random.uniform(low=bounds[k][0],high=bounds[k][1],size=num_data)
	return Z_rand

def multigrid(bounds,Ngrid):
	num_dim = len(bounds)
	Ngrid   -=1
	l_bounds = np.array(bounds)[:,0]
	u_bounds = np.array(bounds)[:,1]
	division = [Ngrid]*num_dim
	grid = UniformBoxGrid(min = l_bounds , max = u_bounds, division = division)
	
	Z = np.zeros(((Ngrid+1)**num_dim,num_dim))
	for k in range(num_dim):
		Z[:,k] = grid.coorv[k].reshape((Ngrid+1)**num_dim)
	return Z




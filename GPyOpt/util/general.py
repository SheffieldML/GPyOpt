import numpy as np

## generates a multidimensional grid uniformly distributes
def samples_multimensional_uniform(num_data,bounds):
        dim = len(bounds)
        Z_rand = np.zeros(shape=(num_data,dim))
        for k in range(0,dim): Z_rand[:,k] = np.random.uniform(low=bounds[k][0],high=bounds[k][1],size=num_data)
	return Z_rand

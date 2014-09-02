#
# The idea of this test is to test the value of the line integral computed
# manually
#
# Javier Gonzalez, September 2014

import GPy
import GPyOpt
from numpy.random import seed
import numpy as np
import matplotlib.pyplot as plt
import scipy

seed(1234) 

# function to have a look it has one global minimum in [1]
def f(X):
	return ((6*X -2)**2)*np.sin(12*X-4) + 10

# bounds
bounds = [(0,1)]

# generate some data
n = 8
x = np.random.uniform(0,1,n)
x_grid = np.arange(0.0, 1.0, 0.01)  
fx = f(x)
f_grid = f(x_grid)
e = np.random.normal(0,0.1,n)  
y = fx + e

# plot of the function and the data
plt.plot(x_grid,f_grid)
plt.plot(x,y,'r.')

# estimate GP model
X = x.reshape(n,1)
Y = y.reshape(n,1)
kernel = GPy.kern.RBF(1, variance=1, lengthscale=1) 
model = GPy.models.GPRegression(X,Y,kernel=kernel)
model.constrain_positive('')
model.optimize_restarts(num_restarts = 10)	
model.optimize('bfgs')
model.plot()

mean, var = model.predict(reshape(x_grid,1))
plt.plot(reshape(x_grid,1),-mean+30)
plt.plot(reshape(x_grid,1),np.sqrt(var))
plt.plot(reshape(x_grid,1),-mean +30 + np.sqrt(var))

## integrals calculated numerically
I_lcb = scipy.integrate.quad(lambda x: LCB(1,model,x), 0, 1)[0]
I_m = scipy.integrate.quad(lambda x: m(model,x), 0, 1)[0]
I_s = scipy.integrate.quad(lambda x: s(model,x), 0, 1)[0]

## add here integrals calculated using the explicit formula



############################### functions we use in the example

def f(X):
        return ((6*X -2)**2)*np.sin(12*X-4)

def reshape(x,input_dim):
	x = np.array(x)
	if len(x)==input_dim: 
		X = x.reshape((1,input_dim))
	else: 
		X = x.reshape((len(x),input_dim)) 
	return X

def get_moments(model,x):
	input_dim = model.input_dim
	x = reshape(x,input_dim)
	m, v = model.predict(x)
	return (m, np.sqrt(v))

def m(model,x):
	x = [x]
        m, s = get_moments(model, x)
 	return -m + 30    ## we add 50 to ensure the function to be positive

def s(model,x):
        x = [x]
        m, s = get_moments(model, x)
        return s

def LCB(par,model,x):
	x = [x]
	m, s = get_moments(model, x) 	
	return -m + par * s + 30 ## we add 50 to ensure the function to be positive





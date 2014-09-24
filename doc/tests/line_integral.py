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
from scipy.special import erfc, erf



seed(1234) 

# bounds
bounds = [(0,1)]

# generate some data
n = 8
x = np.random.uniform(0,1,n)
x_grid = np.arange(0.0, 1.0, 0.01)  
fx = f(x)
f_grid = f(x_grid)
e = np.random.normal(0,0.25,n)  
y = fx + e
y = np.array([100]*n)

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

C = 0 # constant to make the surrogate positive
mean, var = model.predict(reshape(x_grid,1))
plt.plot(reshape(x_grid,1),mean+C)
plt.plot(reshape(x_grid,1),np.sqrt(var))
plt.plot(reshape(x_grid,1),-mean + C + np.sqrt(var))

## integrals calculated numerically
I_lcb = scipy.integrate.quad(lambda x: LCB(1,model,C,x), 0, 1)[0]
I_m = scipy.integrate.quad(lambda x: m(model,C,x), 0, 1)[0]
I_s = scipy.integrate.quad(lambda x: s(model,x), 0, 1)[0]

## add here integrals calculated using the explicit formula


# fix the initial point and the direction (trivial in this case)
x0 = np.array([0]) ## this should be a row vector 1 x input_dim 
r = np.array([1])   ## this should be a row vector 1 x input_dim 
I_bounds = [0,1]

#def integrate_mean(tau,model,x0,r,I_bounds)
tau0 = I_bounds[0]	
tau1 = I_bounds[1]

# from the model we get
X = model.X
theta2 = model.rbf.variance 
sigma2 = model.Gaussian_noise.variance
Lambda_inv =  np.diag(1/(model.rbf.lengthscale**2))
alpha = model.posterior.woodbury_vector

# elements in the integral
a = np.sqrt(np.dot(np.dot(r.T,Lambda_inv),r))
b = np.dot(np.dot(r.T,Lambda_inv),(x0-X.T))
c = np.diag(np.dot(np.dot(X,Lambda_inv),X.T))-2*np.dot(np.dot(X,Lambda_inv),x0)+ np.dot(np.dot(x0,Lambda_inv),x0.T)

# elements of the sum
v = np.exp(-(c-((b**2)/(a**2)))/2)*[erf((a*tau+b)/np.sqrt(2))- erf((a*tau0+b)/np.sqrt(2))]	 
w = np.sqrt(np.pi/2)*theta2*v/a
np.dot(w,alpha)


theta2 * np.exp(-((Lambda_inv* (X[0]-X[1]))**2/2))



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

def m(model,C,x):
	x = [x]
        m, s = get_moments(model, x)
 	return -m + C    ## we add 50 to ensure the function to be positive

def s(model,x):
        x = [x]
        m, s = get_moments(model, x)
        return s

def LCB(par,model,C,x):
	x = [x]
	m, s = get_moments(model, x) 	
	return -m + par * s + C ## we add 50 to ensure the function to be positive





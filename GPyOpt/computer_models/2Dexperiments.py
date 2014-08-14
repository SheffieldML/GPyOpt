from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

'''
Benchmark of optimzation functions. 

List of avaiable functions so far:
- Branin

The classes are oriented to create a python function which contain.
- *.f : the funtion itself
- *.plot: a plot of the function if the dimension is <=2.
- *.sensitivity: The Sobol coefficient per dimension when these are available.
- *.min : value of the global minimum(s) for the default parameters.

NOTE: the imput of .f must be a nxD numpy array. The dimension is calculated within the fucntion.

Javier Gonzalez August, 2014
'''


class function2d:
        def plot(self):
                x1 = np.linspace(-5.0, 10.0, 100)
                x2 = np.linspace(0.0, 15, 100)
                X1, X2 = np.meshgrid(x1, x2)
                X = np.hstack((X1.reshape(100*100,1),X2.reshape(100*100,1)))
                Y = self.f(X)
                fig = plt.figure()
                ax = fig.gca(projection='3d')
                surf = ax.plot_surface(X1, X2, Y.reshape((100,100)), rstride=1, cstride=1, cmap=cm.coolwarm,linewidth=0, antialiased=False)
                ax.zaxis.set_major_locator(LinearLocator(10))
                ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
                plt.show()

class branin(function2d):
	def __init__(self,a=None,b=None,c=None,r=None,s=None,t=None,sd=None):
		self.D = 2
		if a==None: self.a = 1
		else: self.a = a   		
		if b==None: self.b = 5.1/(4*np.pi**2)
		else: self.b = b
		if c==None: self.c = 5/np.pi
                else: self.c = c
		if r==None: self.r = 6
		else: self.r = r
		if s==None: self.s = 10 
		else: self.s = s
		if t==None: self.t = 1/(8*np.pi)
		else: self.t = t	
		if sd==None: self.sd = 0
		else: self.sd=sd
		self.min = ((-np.pi,12.275),(np.pi,2.275),(9.42478,2.475)) 
		self.fmin = 0.397887
	
	def f(self,X):
		if len(X)==self.D: X = X.reshape((1,2))
		n = X.shape[0]
		if X.shape[1] != self.D: 
			return 'Wrong input dimension'  
		else:
			x1 = X[:,0]
			x2 = X[:,1]
			fval = self.a * (x2 - self.b*x1**2 + self.c*x1 - self.r)**2 + self.s*(1-self.t)*np.cos(x1) + self.s 
			if self.sd ==0:
				noise = np.zeros(n).reshape(n,1)
			else:
				noise = np.random.normal(0,self.sd,n).reshape(n,1)
			return fval.reshape(n,1) + noise



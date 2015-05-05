# Copyright (c) 2015, Javier Gonzalez
# Copyright (c) 2015, the GPy Authors (see GPy AUTHORS.txt)
# Licensed under the BSD 3-clause license (see LICENSE.txt)


"""
This is a simple demo to demonstrate the use of Bayesian optimization with GPyOpt with some simple options. Run the example by writing:

import GPyOpt
autoML_demo = GPyOpt.demos.autoML_tuner()

As a result you should see:

- A plot comparing an OLS estimator for a simple regression problem with the obtained parameters using the automatic tuner.
- An object called autoML_dem that contains the results of the optimization process (see reference manual for details). Among the available results you have access to the GP model via


and to the location of the best found location writing.

autoML_demo.x_opt

"""

def autoML_tuner():
    import GPyOpt
    import numpy as np
    import matplotlib.pyplot as plt
    from numpy.random import seed
    seed(123)   
    
    # --- Regression problem
    n = 150
    b0 = 1 
    b1 = 2.5 
    X = np.random.uniform(1,10,n).reshape(n,1)
    Y = b0+b1*X +np.random.normal(0,1.5,n).reshape(n,1)

    # Objective function
    class objective:
        def __init__(self,X,Y):
            self.X = X
            self.Y = Y             
        def f(self,beta):
            return  sum((beta[:,0]+beta[:,1]*self.X-self.Y)**2).reshape(beta.shape[0],1)

                                   
    # --- Problem definition and optimization
    myobj = objective(X,Y)
    bounds = [(0,5),(0,5)]  
    
    # --- Tuning the parameters
    tuner = GPyOpt.methods.autoTune(myobj.f, bounds)   
    beta_tuner = tuner.x_opt             

    # --- OLS solution
    X1 = np.c_[np.ones(n),X ]
    beta_ols   = np.dot(np.dot(np.linalg.inv(np.dot(X1.T,X1)),X1.T),Y)
    
    # --- Plot
    plt.plot(X,Y,'.')
    plt.plot([0,10],[beta_tuner[0],beta_tuner[0]+beta_tuner[1]*10],'k.-',label='GPyOpt')
    plt.plot([0,10],[beta_ols[0],beta_ols[0]+beta_ols[1]*10],label='OLS')
    plt.legend()

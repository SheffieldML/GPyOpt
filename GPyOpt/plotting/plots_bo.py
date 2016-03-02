# Copyright (c) 2015, Javier Gonzalez
# Copyright (c) 2015, the GPy Authors (see GPy AUTHORS.txt)
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from pylab import grid
import matplotlib.pyplot as plt
from pylab import savefig


def plot_acquisition(bounds,input_dim,model,Xdata,Ydata,acquisition_function,suggested_sample, filename = None):
    '''
    Plots of the model and the acquisition function in 1D and 2D examples.
    '''
      
    # Plots in dimension 1
    if input_dim ==1:
        X = np.arange(bounds[0][0], bounds[0][1], 0.001)
        X = X.reshape(len(X),1)
        acqu = acquisition_function(X)
        acqu_normalized = (-acqu - min(-acqu))/(max(-acqu - min(-acqu))) # normalize acquisition 
        m, v = model.predict(X.reshape(len(X),1))
        plt.ioff()        
        plt.figure(figsize=(10,5)) 
        plt.subplot(2, 1, 1)
        plt.plot(X, m, 'b-', label=u'Posterior mean',lw=2)
        plt.fill(np.concatenate([X, X[::-1]]), \
                np.concatenate([m - 1.9600 * np.sqrt(v),
                            (m + 1.9600 * np.sqrt(v))[::-1]]), \
                alpha=.5, fc='b', ec='None', label='95% C. I.') 
        plt.plot(X, m-1.96*np.sqrt(v), 'b-', alpha = 0.5)
        plt.plot(X, m+1.96*np.sqrt(v), 'b-', alpha=0.5)     
        plt.plot(Xdata, Ydata, 'r.', markersize=10, label=u'Observations')
        plt.axvline(x=suggested_sample[len(suggested_sample)-1],color='r')
        plt.title('Model and observations')
        plt.ylabel('Y')
        plt.xlabel('X')
        plt.legend(loc='upper left')
        plt.xlim(*bounds)
        grid(True)  
        plt.subplot(2, 1, 2)
        plt.axvline(x=suggested_sample[len(suggested_sample)-1],color='r')
        plt.plot(X,acqu_normalized, 'r-',lw=2) 
        plt.xlabel('X')
        plt.ylabel('Acquisition value')
        plt.title('Acquisition function')
        grid(True)
        plt.xlim(*bounds)
        if filename!=None:
            savefig(filename)
        else:
            plt.show()

    if input_dim ==2:
        X1 = np.linspace(bounds[0][0], bounds[0][1], 200)
        X2 = np.linspace(bounds[1][0], bounds[1][1], 200)
        x1, x2 = np.meshgrid(X1, X2)
        X = np.hstack((x1.reshape(200*200,1),x2.reshape(200*200,1)))
        acqu = acquisition_function(X)
        acqu_normalized = (-acqu - min(-acqu))/(max(-acqu - min(-acqu)))
        acqu_normalized = acqu_normalized.reshape((200,200))
        m, v = model.predict(X) 
        plt.figure(figsize=(15,5))
        plt.subplot(1, 3, 1)            
        plt.contourf(X1, X2, m.reshape(200,200),100)
        plt.plot(Xdata[:,0], Xdata[:,1], 'r.', markersize=10, label=u'Observations')
        plt.colorbar()  
        plt.xlabel('X1')
        plt.ylabel('X2')            
        plt.title('Posterior mean')
        plt.axis((bounds[0][0],bounds[0][1],bounds[1][0],bounds[1][1]))
        ##
        plt.subplot(1, 3, 2)
        plt.plot(Xdata[:,0], Xdata[:,1], 'r.', markersize=10, label=u'Observations')
        plt.contourf(X1, X2, np.sqrt(v.reshape(200,200)),100)
        plt.colorbar()
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.title('Posterior sd.')
        plt.axis((bounds[0][0],bounds[0][1],bounds[1][0],bounds[1][1]))
        ##
        plt.subplot(1, 3, 3)
        plt.contourf(X1, X2, acqu_normalized,100)
        plt.colorbar()
        plt.plot(suggested_sample[:,0],suggested_sample[:,1],'k.', markersize=10)
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.title('Acquisition function')
        plt.axis((bounds[0][0],bounds[0][1],bounds[1][0],bounds[1][1]))
        if filename!=None:savefig(filename)


def plot_convergence(Xdata,best_Y,s_in_min, filename = None):
    '''
    Plots to evaluate the convergence of standard Bayesian optimization algorithms
    '''
    n = Xdata.shape[0]  
    aux = (Xdata[1:n,:]-Xdata[0:n-1,:])**2      
    distances = np.sqrt(aux.sum(axis=1))

    ## Distances between consecutive x's
    plt.figure(figsize=(15,5))
    plt.subplot(1, 3, 1)
    plt.plot(range(n-1), distances, '-ro')
    plt.xlabel('Iteration')
    plt.ylabel('d(x[n], x[n-1])')
    plt.title('Distance between consecutive x\'s')
    grid(True)

    # Estimated m(x) at the proposed sampling points
    plt.subplot(1, 3, 2)
    plt.plot(range(n),best_Y,'-o')
    plt.title('Value of the best selected sample')
    plt.xlabel('Iteration')
    plt.ylabel('Best y')
    grid(True)

    # Plot of the proposed v(x) at the proposed sampling points
    plt.subplot(1, 3, 3)
    plt.errorbar(range(n),[0]*n , yerr=s_in_min[:,0],ecolor='b', capthick=1)
    plt.title('Predicted sd. in the next sample')
    plt.xlabel('Iteration')
    plt.ylim(0,max(s_in_min[:,0])+np.sqrt(max(s_in_min[:,0])))
    plt.ylabel('CI (centered at zero)')
    grid(True)
    if filename!=None:
        savefig(filename)
    else:
        plt.show()


    
    

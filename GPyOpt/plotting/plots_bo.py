import numpy as np
from pylab import plot, xlabel, ylabel, title, grid
import matplotlib.pyplot as plt
from pylab import savefig

from ..util.general import ellipse, best_value

def plot_acquisition(bounds,input_dim,model,Xdata,Ydata,acquisition_function,suggested_sample, filename = None):
	'''
	Plots the model and the acquisition function in 1D and 2D examples
	'''
	if input_dim ==1:
		X = np.arange(bounds[0][0], bounds[0][1], 0.001)
       		acqu = acquisition_function(X)
		acqu_normalized = (-acqu - min(-acqu))/(max(-acqu - min(-acqu))) # normalize acq in (0,1)
		m, v = model.predict(X.reshape(len(X),1))
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
		plt.axvline(x=suggested_sample,color='r')
		plt.title('Model and observations')
		plt.ylabel('Y')
		plt.xlabel('X')
		plt.legend(loc='upper left')
                plt.xlim(*bounds)
		grid(True)	
		plt.subplot(2, 1, 2)
		plt.axvline(x=suggested_sample,color='r')
		plt.plot(X,acqu_normalized, 'r-',lw=2) 
		plt.xlabel('X')
		plt.ylabel('Acquisition value')
		plt.title('Acquisition function')
		grid(True)
                plt.xlim(*bounds)

	if input_dim ==2:
		X1 = np.linspace(bounds[0][0], bounds[0][1], 200)
		X2 = np.linspace(bounds[1][0], bounds[1][1], 200)
		x1, x2 = np.meshgrid(X1, X2)
		X = np.hstack((x1.reshape(200*200,1),x2.reshape(200*200,1)))
		acqu = acquisition_function(X)
		acqu_normalized = (-acqu - min(-acqu))/(max(-acqu - min(-acqu)))
		acqu_normalized = acqu_normalized.reshape((200,200))
		m, v = model.predict(X)	
		#
		eX3, eY3 = ellipse(Xdata,nstd=3)
		eX2, eY2 = ellipse(Xdata,nstd=2)
		eX1, eY1 = ellipse(Xdata,nstd=1)
		pos = X.mean(axis=0)
			##
		plt.figure(figsize=(15,5))
		plt.subplot(1, 3, 1)			
		plt.contourf(X1, X2, m.reshape(200,200),100)
		plt.plot(Xdata[:,0], Xdata[:,1], 'r.', markersize=10, label=u'Observations')
		plt.colorbar()
		#plt.plot(eX1,eY1,"k.-",ms=1,lw=3,alpha = 0.9)
		#plt.plot(eX2,eY2,"k.-",ms=1,lw=3,alpha = 0.6)
		#plt.plot(eX3,eY3,"k.-",ms=1,lw=3,alpha = 0.3)		
		plt.xlabel('X1')
		plt.ylabel('X2')			
		plt.title('Posterior mean')
		plt.axis((bounds[0][0]-0.5,bounds[0][1]+0.5,bounds[1][0]-0.5,bounds[1][1]+0.5))
		##
		plt.subplot(1, 3, 2)
		plt.plot(Xdata[:,0], Xdata[:,1], 'r.', markersize=10, label=u'Observations')
		plt.contourf(X1, X2, np.sqrt(v.reshape(200,200)),100)
		plt.colorbar()
		#plt.plot(eX1,eY1,"k.-",ms=1,lw=3,alpha = 0.9)
		#plt.plot(eX2,eY2,"k.-",ms=1,lw=3,alpha = 0.6)
		#plt.plot(eX3,eY3,"k.-",ms=1,lw=3,alpha = 0.3)
		plt.xlabel('X1')
		plt.ylabel('X2')
		plt.title('Posterior sd.')
		plt.axis((bounds[0][0]-0.5,bounds[0][1]+0.5,bounds[1][0]-0.5,bounds[1][1]+0.5))
		##
		plt.subplot(1, 3, 3)
		plt.contourf(X1, X2, acqu_normalized,100)
		plt.colorbar()
		#plt.plot(eX1,eY1,"k.-",ms=1,lw=3,alpha = 0.9)
		#plt.plot(eX2,eY2,"k.-",ms=1,lw=3,alpha = 0.6)
		#plt.plot(eX3,eY3,"k.-",ms=1,lw=3,alpha = 0.3)
		plt.plot(suggested_sample[0],suggested_sample[1],'k.', markersize=10)
		plt.xlabel('X1')
		plt.ylabel('X2')
		plt.title('Acquisition function')
		plt.axis((bounds[0][0]-0.5,bounds[0][1]+0.5,bounds[1][0]-0.5,bounds[1][1]+0.5))
		#plt.subtitle((np.around(suggested_sample,3))




def plot_convergence(Xdata,best_Y,s_in_min):
	'''
	Plots three plots to evaluate the convergence of the algorithm
	'''
	n = Xdata.shape[0]	
	aux = (Xdata[1:n,:]-Xdata[0:n-1,:])**2		
	distances = np.sqrt(aux.sum(axis=1))
	## plot of distances between consecutive x's
	plt.figure(figsize=(15,5))
	plt.subplot(1, 3, 1)
	plt.plot(range(n-1), distances, '-ro')
	plt.xlabel('Iteration')
	plt.ylabel('d(x[n], x[n-1])')
	plt.title('Distance between consecutive x\'s')
	grid(True)
	# plot of the estimated m(x) at the proposed sampling points
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



	
	

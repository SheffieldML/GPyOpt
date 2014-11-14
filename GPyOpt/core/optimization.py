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

#def density_sampling_optimization(acquisition_function, bounds, model,Ninit):
#	    mins = np.zeros((Ninit,len(bounds)))
#		fmins = np.zeros(Ninit)
#	    samples = ## sample here from the density
#       for k in range(Ninit):
#          res = scipy.optimize.minimize(acquisition_function, x0 = samples[k,:] ,method='SLSQP',bounds=bounds)
#           mins[k] = res.x
#           fmins[k] = res.fun
#    return mins[np.argmin(fmins)]

###
### function for running the density sample function
###
'''
def constant_line_lcb(model,z,x0,r,tau0,tau1,C,sign=-1):
	return scipy.integrate.quad(lambda tau: line_unnorm_lcb(model,z,x0,r,tau,C,sign=-1), tau0, tau1)[0]

def line_unnorm_lcb(model,z,x0,r,tau,C,sign=-1):
	m, s = get_moments(model, x0+tau*r)
	return -m + z * s + C
			
def line_norm_lcb(model,z,x0,r,tau0,tau1,tau,C,sign=-1):
	cte = constant_line_lcb(model,z,x0,r,tau0,tau1,C,sign)
	m, s = get_moments(model, x0+tau*r)
	return (-m + z * s + C)/cte

def cum_line_norm(model,z,x0,r,tau0,tau1,tau,C,sign=-1):
	cte = scipy.integrate.quad(lambda x: line_unnorm_lcb(model,z,x0,r,x,C,sign=-1), tau0, tau1)[0]
	cumulative = scipy.integrate.quad(lambda x: line_unnorm_lcb(model,z,x0,r,x,C,sign=-1), tau0, tau)[0]
	return cumulative/cte

def generate_samples_line(N,model,bounds,z,x0,r,tau0,tau1,sign=-1):
	X = model.X
	C = max(model.predict(X)[0])
	samples  = np.zeros((N,X.shape[1]))

	for i in range(N):
		U = random()
		f = lambda tau: (cum_line_norm(model=model,z=z,x0=x0,r=r,tau0=tau0,tau1=tau1,tau=tau, C=C,sign=sign) - U)**2
		res = minimize_scalar(f,bounds=(tau0,tau1), method='bounded')
		samples[i,:] = x0 + r*res.x
	return samples

def genererte_line(model,bounds,z,sign):
	input_dim = len(bounds)
	X = model.X
	C = max(model.predict(X)[0])
	pos = X.mean(axis=0)
	cov = np.cov(X, rowvar=False) 	
	x0 =  np.random.multivariate_normal(pos, cov,1)

	while x0[0,0]<bounds[0][0] or x0[0,0]>bounds[0][1] or  x0[0,1]<bounds[1][0] or x0[0,1]>bounds[1][1]:
		x0 =  np.random.multivariate_normal(pos, cov,1)
		r0 =  np.random.multivariate_normal(pos, cov,1)
		r = r0/np.linalg.norm(r0) 
		tau0,tau1 = get_limits(x0,r,bounds)
		integral = constant_line_lcb(model,z,x0,r,tau0,tau1,C,sign=sign)
	return (x0,r,tau0,tau1,integral)


#TODO extend for functions with arbitrary number of dimensions
def get_limits(x0,r,bounds):
	input_dim = len(bounds)
	res = np.zeros((input_dim*2,3))

	# for the bouds for x1
	res[0,0] = bounds[0][0] 
	res[0,2] = (bounds[0][0]-x0[0,0])/r[0][0]
	res[0,1] = x0[0][1] + res[0,2]*r[0][1]

	res[1,0] = bounds[0][1]
	res[1,2] = (bounds[0][1]-x0[0,0])/r[0][0]
	res[1,1] = x0[0][1] + res[1,2]*r[0][1]

	# for the bouds for x2
	res[2,1] = bounds[1][0]
	res[2,2] = (bounds[1][0]-x0[0,1])/r[0][1]
	res[2,0] = x0[0][0] + res[2,2]*r[0][0]

	res[3,1] = bounds[1][1]
	res[3,2] = (bounds[1][1]-x0[0,1])/r[0][1]
	res[3,0] = x0[0][0] + res[3,2]*r[0][0]

	## select 
	log1 = (res[:,0]>=bounds[0][0]).astype(int) 
	log2 = (res[:,0]<=bounds[0][1]).astype(int)
	log3 = (res[:,1]>=bounds[1][0]).astype(int) 
	log4 = (res[:,1]<=bounds[1][1]).astype(int)

	sel = (log1 + log2 +log3 +log4) == 4
	tau0 = min(res[sel,2])
	tau1 = max(res[sel,2])
	return (tau0,tau1)


def generate_initial_points(model, bounds,z,sign=-1, Ndir = 100, percentile = 90, Nsamples = 5):
	input_dim = model.X.shape[1]

	# generate Ndir directions and save the value of the integral
	m_x0 = np.zeros((Ndir,input_dim))
	m_r = np.zeros((Ndir,input_dim))
	v_tau0 = np.zeros(Ndir)
	v_tau1 = np.zeros(Ndir)
	v_int = np.zeros(Ndir)

	for k in range(Ndir):
		m_x0[k,:],m_r[k,:],v_tau0[k],v_tau1[k],v_int[k] = genererte_line(model,bounds,z,sign)	

	m_x0 = m_x0[v_int > np.percentile(v_int,percentile),:]
	m_r = m_r[v_int > np.percentile(v_int,percentile),:]
	v_tau0 = v_tau0[v_int > np.percentile(v_int,percentile)]
	v_tau1 = v_tau1[v_int > np.percentile(v_int,percentile)]
	v_int = v_int[v_int > np.percentile(v_int,percentile)]

	###
	### select samples from the selected directions
	###
	samples = np.array([]).reshape(0,input_dim)

	for j in range(len(v_int)):
		new_samples = generate_samples_line(Nsamples,model,bounds,z,m_x0[j,:],m_r[j,:],v_tau0[j],v_tau1[j],sign=-1)
		samples = np.vstack((samples,new_samples))

	return samples
'''
















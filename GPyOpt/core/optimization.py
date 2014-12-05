import numpy as np
from ..util.general import multigrid, samples_multidimensional_uniform, reshape
from scipy.stats import norm
import scipy
import GPyOpt


def random_batch_optimization(acquisition, bounds, acqu_optimize_restarts, acqu_optimize_method, model, n_inbatch):
    '''

    '''
    X_batch = optimize_acquisition(acquisition, bounds, acqu_optimize_restarts, acqu_optimize_method, model)
    k=1 
    while k<n_inbatch:
        new_sample = samples_multidimensional_uniform(bounds,1)
        X_batch = np.vstack((X_batch,new_sample))  
        k +=1
    return X_batch


def adaptive_batch_optimization(acquisition, bounds, acqu_optimize_restarts, acqu_optimize_method, model, n_inbatch, alpha_L, alpha_Min):
    '''
    Computes batch optimzation using by acquisition penalization
    :param acquisition: acquisition function in which the batch selection is based
    :param bounds: the box constrains of the optimization
    :restarts: the number of restarts in the optimization of the surrogate
    :method: the method to optimize the aquisition function
    :model: the GP model based on the current samples
    :n_inbatch: the number of samples to collect
    :alpha_L: z quantile for the estimation of the lipchiz constant L
    :alpha_Min: z quantile for the estimation of the minimum Min
    '''
    X_batch = optimize_acquisition(acquisition, bounds, acqu_optimize_restarts, acqu_optimize_method, model, X_batch=None, L=None, Min=None)
    k=1
    if n_inbatch>1:
        L = estimate_L(model,bounds,alpha_L)
       # print 'L is'
       # print  L
        Min = estimate_Min(model,bounds,alpha_Min)
       # print 'Min is'
       # print Min

    while k<n_inbatch:
        new_sample = optimize_acquisition(acquisition, bounds, acqu_optimize_restarts, acqu_optimize_method, model, X_batch, L, Min)
        X_batch = np.vstack((X_batch,new_sample))  
        k +=1
    return X_batch


# TODO Estimates 'the lipchitz constant' of a model. Note that we need to use the gradients here. The lipchiz constant is bounded by the maximum derivative.
def estimate_L(model,bounds,alpha=0.025):
    def df(x,model,alpha):
        x = reshape(x,model.X.shape[1])
        dmdx, dsdx = model.predictive_gradients(x)
        res = np.sqrt((dmdx*dmdx).sum(1)) #+ norm.ppf(1-alpha)*dsdx
        return -res
   
    samples = samples_multidimensional_uniform(bounds,25)
    pred_samples = df(samples,model,alpha)
    x0 = samples[np.argmin(pred_samples)]
    #print x0
    #print df(x0,model,alpha)
    minusL = scipy.optimize.minimize(df,x0, method='SLSQP',bounds=bounds, args = (model,alpha)).fun[0][0]
    L = -minusL
    if L< 0.1: L =100  ## to avoid problems in cases in which the mode is flat
    return L

# Estimates 'the minimum' of a model
def estimate_Min(model,bounds,alpha=0.025):
    def f(x,model,alpha):
        if len(x.flatten())==2:
            x = x.reshape(1,2)
        m,v = model.predict(x)
        res = m #+ norm.ppf(1-alpha)*np.sqrt(abs(v))
        return res
    samples = samples_multidimensional_uniform(bounds,25)
    pred_samples = f(samples,model,alpha)
    x0 = samples[np.argmin(pred_samples)]
    return scipy.optimize.minimize(f,x0, method='SLSQP',bounds=bounds, args = (model,alpha)).fun[0][0]

# creates the function to define the esclusion zones
def hammer_function(x,x0,L,Min,model):
    x0 = x0.reshape(1,len(x0))
    m = model.predict(x0)[0]
    s = np.sqrt(model.predict(x0)[1])
    r_x0 = (m-Min)/L
    s_x0 = s/L
    return (norm.cdf((np.sqrt(((x-x0)**2).sum(1))- r_x0)/s_x0)).T
    #return (norm.cdf((np.sqrt(((x-x0)**2).sum(1))- r_x0)/s_x0)-norm.cdf(-r_x0/s_x0)).T


# creates a penalized acquisition function using 'hammer' functions around the points collected in the batch
def penalized_acquisition(x, acquisition, bounds, model, X_batch=None, L=None, Min=None):
    sur_min = min(-acquisition(model.X))  # assumed minimum of the minus acquisition
    fval = -acquisition(x)-np.sign(sur_min)*(abs(sur_min)) 
    if X_batch!=None:
        X_batch = reshape(X_batch,2)
        for i in range(X_batch.shape[0]):            
            fval = np.multiply(fval,hammer_function(x, X_batch[i,], L, Min, model))
    return -fval

### Optimization of the aquisition function
def optimize_acquisition(acquisition, bounds, acqu_optimize_restarts, acqu_optimize_method, model, X_batch=None, L=None, Min=None):
    if acqu_optimize_method=='brute':
        res = full_acquisition_optimization(acquisition,bounds,acqu_optimize_restarts, model, 'brute', X_batch, L, Min)
    elif acqu_optimize_method=='random':
        res =  full_acquisition_optimization(acquisition,bounds,acqu_optimize_restarts, model, 'random', X_batch, L, Min)
    elif acqu_optimize_method=='fast_brute':
        res =  fast_acquisition_optimization(acquisition,bounds,acqu_optimize_restarts, model, 'brute', X_batch, L, Min)
    elif acqu_optimize_method=='fast_random':
        res =  fast_acquisition_optimization(acquisition,bounds,acqu_optimize_restarts, model, 'random', X_batch, L, Min)
    return res

### optimizes the acquisition function using a local optimizer in the best point
def fast_acquisition_optimization(acquisition, bounds,acqu_optimize_restarts, model, method_type, X_batch=None, L=None, Min=None):
    if method_type=='random':
                samples = samples_multidimensional_uniform(bounds,acqu_optimize_restarts)
    else:
        samples = multigrid(bounds, acqu_optimize_restarts)
    pred_samples = acquisition(samples)
    x0 =  samples[np.argmin(pred_samples)]
    res = scipy.optimize.minimize(penalized_acquisition, x0=np.array(x0),method='SLSQP',bounds=bounds, args=(acquisition, bounds, model, X_batch, L, Min))
    return res.x

### optimizes the acquisition function by taking the best of a number of local optimizers
def full_acquisition_optimization(acquisition, bounds, acqu_optimize_restarts, model, method_type, X_batch=None, L=None, Min=None):
    if method_type=='random':
        samples = samples_multidimensional_uniform(bounds,acqu_optimize_restarts)
    else:
        samples = multigrid(bounds, acqu_optimize_restarts)
    mins = np.zeros((acqu_optimize_restarts,len(bounds)))
    fmins = np.zeros(acqu_optimize_restarts)
    for k in range(acqu_optimize_restarts):
        res = scipy.optimize.minimize(penalized_acquisition, x0 = samples[k,:] ,method='SLSQP',bounds=bounds, args=(acquisition, bounds, model, X_batch, L, Min))
        mins[k] = res.x
        fmins[k] = res.fun
    return mins[np.argmin(fmins)]


def hybrid_batch_optimization(acqu_name, acquisition_par, acquisition, bounds, acqu_optimize_restarts, acqu_optimize_method, model, n_inbatch):
    
    model_copy = model.copy()
    X = model_copy.X 
    Y = model_copy.Y
    input_dim = X.shape[1] 
    kernel = model_copy.kern
    
    X_new = optimize_acquisition(acquisition, bounds, acqu_optimize_restarts, acqu_optimize_method, model, X_batch=None, L=None, Min=None)
    X_batch = reshape(X_new,input_dim)

    k=1
    while k<n_inbatch:
        X = np.vstack((X,reshape(X_new,input_dim)))       # update the sample
        Y = np.vstack((Y,model.predict(reshape(X_new, input_dim))[0]))
        
        batchBO = GPyOpt.methods.BayesianOptimization(f=0, 
                                    bounds= bounds, 
                                    X=X, 
                                    Y=Y, 
                                    kernel = kernel,
                                    acquisition = acqu_name, 
                                    acquisition_par = acquisition_par)
        
        batchBO.start_optimization(max_iter = 0, 
                                    n_inbatch=1, 
                                    acqu_optimize_method = acqu_optimize_method,  
                                    acqu_optimize_restarts = acqu_optimize_restarts, 
                                    stop_criteria = 1e-6)
        
        X_new = batchBO.suggested_sample
        X_batch = np.vstack((X_batch,X_new))
        model_batch = batchBO.model
        k+=1    
    return X_batch










'''
def fast_surrogate_optimization(acquisition_function, bounds, n_init, method='random'):
	if method=='random':#
                samples = samples_multidimensional_uniform(bounds,n_init)
        else:
		samples = multigrid(bounds, n_init)
	pred_samples = acquisition_function(samples)
	x0 =  samples[np.argmin(pred_samples)]
	res = scipy.optimize.minimize(acquisition_function, x0=np.array(x0),method='SLSQP',bounds=bounds)
	return res.x

def surrogate_optimization(acquisition_function, bounds, n_init, method='random'):
	if method=='random':
		samples = samples_multidimensional_uniform(bounds,n_init)
	else:
		samples = multigrid(bounds, n_init)	
	mins = np.zeros((n_init,len(bounds)))
	fmins = np.zeros(n_init)
	for k in range(n_init):
		res = scipy.optimize.minimize(acquisition_function, x0 = samples[k,:] ,method='SLSQP',bounds=bounds)
		mins[k] = res.x
		fmins[k] = res.fun
	return mins[np.argmin(fmins)]

def batch_optimization(self):
	This function merges the different approaches for optimizing the acquisition function with a batch optimization in which
	points within the batch are collected using the prrdictive mean of the model to generate sequential new data points.
	
	if self.acqu_optimize_method=='brute':
		X_batch = surrogate_optimization(self.acquisition_func.acquisition_function, self.bounds,self.acqu_optimize_restarts, method='brute')
	elif self.acqu_optimize_method=='random':
		X_batch =  surrogate_optimization(self.acquisition_func.acquisition_function,self.bounds,self.acqu_optimize_restarts, method='random')
	elif self.acqu_optimize_method=='fast_brute':
                X_batch =  fast_surrogate_optimization(self.acquisition_func.acquisition_function,self.bounds,self.acqu_optimize_restarts, method='brute')
	elif self.acqu_optimize_method=='fast_random':
                X_batch =  fast_surrogate_optimization(self.acquisition_func.acquisition_function,self.bounds,self.acqu_optimize_restarts, method='random')
	else:
		print 'Wrong aquisition optimizer inserted.'
	k=1 
	while k<self.n_inbatch:
		model_batch = self.model
		X = np.vstack((self.X,X_batch))
		Y = np.vstack((self.Y,model_batch.predict(reshape(X_batch,self.input_dim))[0]))
		auxBO = GPyOpt.methods.BayesianOptimization(f=self.f, bounds=self.bounds,X=X,Y=Y,kernel=self.kernel, acquisition = self.acqu_name, normalize = self.normalize)
		auxBO.start_optimization(max_iter = 0, acqu_optimize_restarts = 20)			
		X_batch = np.vstack((X_batch,auxBO.suggested_sample))
		k+=1
	return X_batch


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
















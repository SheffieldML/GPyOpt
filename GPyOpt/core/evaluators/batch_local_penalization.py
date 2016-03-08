from .base import EvaluatorBase

class LocalPenalization(EvaluatorBase):
    """
    Class for Expected improvement acquisition functions.
    """
    def __init__(self, acquisition, batch_size):
        super(LocalPenalization, self).__init__(acquisition, batch_size)
        self.acquisition = acquisition
        self.batch_size = batch_size
        self.normalize_Y = normalize_Y

    def compute_batch(self):
    	pass



###########
###########  ----  Old stuff
###########



#     '''
#     Computes batch optimization using by acquisition penalization using Lipschitz inference.

#     :param acquisition: acquisition function in which the batch selection is based
#     :param d_acquisition: gradient of the acquisition
#     :param bounds: the box constrains of the optimization
#     :param acqu_optimize_restarts: the number of restarts in the optimization of the surrogate
#     :param acqu_optimize_method: the method to optimize the acquisition function
#     :param model: the GP model based on the current samples
#     :param n_inbatch: the number of samples to collect
#     '''
#     from .acquisition import AcquisitionMP
#     assert isinstance(acquisition, AcquisitionMP)
#     acq_func = acquisition.acquisition_function
#     d_acq_func = acquisition.d_acquisition_function

#     acquisition.update_batches(None,None,None)    
#     # Optimize the first element in the batch
#     X_batch = optimize_acquisition(acq_func, d_acq_func, bounds, acqu_optimize_restarts, acqu_optimize_method, model, X_batch=None, L=None, Min=None)
#     k=1
#     #d_acq_func = None  # gradients are approximated  with the batch. 
    
#     if n_inbatch>1:
#         # ---------- Approximate the constants of the the method
#         L = estimate_L(model,bounds)
#         Min = estimate_Min(model,bounds)

#     while k<n_inbatch:
#         acquisition.update_batches(X_batch,L,Min)
#         # ---------- Collect the batch (the gradients of the acquisition are approximated for k =2,...,n_inbatch)
#         new_sample = optimize_acquisition(acq_func, d_acq_func, bounds, acqu_optimize_restarts, acqu_optimize_method, model, X_batch, L, Min)
#         X_batch = np.vstack((X_batch,new_sample))
#         k +=1
#     acquisition.update_batches(None,None,None)
#     return X_batch



#   def estimate_L(model,bounds,storehistory=True):
#     '''
#     Estimate the Lipschitz constant of f by taking maximizing the norm of the expectation of the gradient of *f*.
#     '''
#     def df(x,model,x0):
#         x = reshape(x,model.X.shape[1])
#         dmdx,_ = model.predictive_gradients(x)
#         res = np.sqrt((dmdx*dmdx).sum(1)) # simply take the norm of the expectation of the gradient
#         return -res
   
#     samples = samples_multidimensional_uniform(bounds,500)
#     samples = np.vstack([samples,model.X])
#     pred_samples = df(samples,model,0)
#     x0 = samples[np.argmin(pred_samples)]
#     res = scipy.optimize.minimize(df,x0, method='L-BFGS-B',bounds=bounds, args = (model,x0), options = {'maxiter': 200})
#     minusL = res.fun[0][0]
#     L = -minusL
#     if L<1e-7: L=10  ## to avoid problems in cases in which the model is flat.
#     return L


# def estimate_Min(model,bounds):
#     '''
#     Takes the estimated minimum as the minimum value in the sample. (this function is now nonsense but we will used in further generalizations)
    
#     '''
#     return model.Y.min()
# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from .base import EvaluatorBase
from ...util.stats import initial_design
import numpy as np
from ...models import GPModel
from ..bo import BO
from ..task.objective import SingleObjective
from ..evaluators import Sequential
from copy import deepcopy

#### 
#### TODO! : runs but it does not work well yet!!! THERE IS SOMETHING WRONG
####


class Predictive(EvaluatorBase):
    """
    Class a predictive batch method. Computes the element of the batch Sequentially by predicting outputs of the function with the mean of the GP.
    The model is updated after every fantasized evaluation is collected.

    :param acquisition: acquisition function to be used to compute the batch.
    :param batch size: the number of elements in the batch.
    :normalize_Y: whether to normalize the outputs.

    """
    def __init__(self, acquisition, batch_size, normalize_Y):
        super(Predictive, self).__init__(acquisition, batch_size)
        self.normalize_Y = normalize_Y

    def compute_batch(self):
        """
        Computes the elements of the batch sequentially by using predictions from the model.
        """

        # --- Compute local entities
        X = self.acquisition.model.model.X.copy()
        Y = self.acquisition.model.model.Y.copy()
        model_local = self.acquisition.model.copy()  #### how to make a copy of this???
        space_local = self.acquisition.space
        objective_local = SingleObjective(lambda x: 0 )
        acquisition_local = deepcopy(self.acquisition) 
        collection_method = Sequential(acquisition_local,1)

        # --- Optimize the acquisition by first time
        X_new = np.atleast_2d(self.acquisition.optimize())
        X_batch = X_new
        k = 1

        # --- Collect the rest of the elements in the batch by 
        while k<self.batch_size:
            X = np.vstack((X,X_new))       # update the sample within the batch
            Y = np.vstack((Y,model_local.predict(X_new)[0]))
            print((X,Y))
            model_local.updateModel(X,Y,None,None)
            print((model_local.model.X, model_local.model.Y))

            print(X)
            try: # this exception is included in case two equal points are selected in a batch, in this case the method stops
                batchBO = BO(model_local, space_local, objective_local, acquisition_local, collection_method, X_init=X.copy(), Y_init=Y.copy(), normalize_Y = self.normalize_Y)  
            except np.linalg.linalg.LinAlgError:
                print('Optimization stopped. Two equal points selected.')
                break        

            batchBO.run_optimization(max_iter = 0)        
            X_new = batchBO.suggested_sample
            X_batch = np.vstack((X_batch,X_new))
            k+=1    

        return X_batch




# def predictive_batch_optimization(acqu_name, acquisition_par, acquisition, d_acquisition, bounds, acqu_optimize_restarts, acqu_optimize_method, model, n_inbatch):   
#     '''
#     Computes batch optimization using the predictive mean to obtain new batch elements

#     :param acquisition: acquisition function in which the batch selection is based
#     :param d_acquisition: gradient of the acquisition
#     :param bounds: the box constrains of the optimization
#     :param acqu_optimize_restarts: the number of restarts in the optimization of the surrogate
#     :param acqu_optimize_method: the method to optimize the acquisition function
#     :param model: the GP model based on the current samples
#     :param n_inbatch: the number of samples to collect
#     '''
#     model_copy = model.copy()
#     X = model.X.copy() 
#     Y = model.Y.copy()
#     input_dim = X.shape[1] 
#     #kernel = model_copy.kern    

#     # Optimization of the first element in the batch
#     X_new = optimize_acquisition(acquisition, d_acquisition, bounds, acqu_optimize_restarts, acqu_optimize_method, model, X_batch=None, L=None, Min=None)
#     X_batch = reshape(X_new,input_dim)
#     k=1
#     while k<n_inbatch:
#         X = np.vstack((X,reshape(X_new,input_dim)))       # update the sample within the batch
#         Y = np.vstack((Y,model.predict(reshape(X_new, input_dim))[0]))
       
#         try: # this exception is included in case two equal points are selected in a batch, in this case the method stops
#             batchBO = GPyOpt.methods.BayesianOptimization(f=0, 
#                                         bounds= bounds, 
#                                         X=X, 
#                                         Y=Y, 
#                                         kernel = kernel,
#                                         acquisition = acqu_name, 
#                                         acquisition_par = acquisition_par)
#         except np.linalg.linalg.LinAlgError:
#             print('Optimization stopped. Two equal points selected.')
#             break        

#         batchBO.run_optimization(max_iter = 0, 
#                                     n_inbatch=1, 
#                                     acqu_optimize_method = acqu_optimize_method,  
#                                     acqu_optimize_restarts = acqu_optimize_restarts, 
#                                     eps = 1e-6,verbose = False)
        
#         X_new = batchBO.suggested_sample
#         X_batch = np.vstack((X_batch,X_new))
#         k+=1    
#     return X_batch

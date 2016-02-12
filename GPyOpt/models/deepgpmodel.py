# Copyright (c) 2015, Javier Gonzalez
# Copyright (c) 2015, the GPy Authors (see GPy AUTHORS.txt)
# Licensed under the BSD 3-clause license (see LICENSE.txt)

## TODO

# from .base import BOModel
# import numpy as np
# import GPy

# #
# # ------- TODO
# #

# class DeepGPModel(BOModel):

#     analytical_gradient_prediction = True
    
#     def __init__(self, XXX):

#         pass            

#     def _create_model(self, X, Y):
#         '''
#         Initializes a deep Gaussian Process with one hidden layer over *f*.
#         :param X: input observations.
#         :param Y: output values.
#         '''

#         import socket
#         self.useGPU = False
#         if socket.gethostname()[0:4] == 'node':
#             print 'Using GPU!'
#             self.useGPU = True

#         # --- kernel and dimension of the hidden layer
#         self.Ds = 1
#         kern = [GPy.kern.Matern32(self.Ds, ARD=False), GPy.kern.Matern32(self.X.shape[1], ARD=False)]

#         if self.model_type == 'deepgp_back_constraint':
#             self.model = deepgp.DeepGP([self.Y.shape[1],self.Ds, self.X.shape[1]], self.Y, X=X, num_inducing=self.num_inducing, kernels=kern, MLP_dims=[[100,50],[]],repeatX=True)
        
#         elif self.model_type == 'deepgp':
#             self.model = deepgp.DeepGP([self.Y.shape[1],self.Ds, self.X.shape[1]], self.Y, X=X, num_inducing=self.num_inducing, kernels=kern, back_constraint=False,repeatX=True)

#         if self.exact_feval == True:
#             self.model.obslayer.Gaussian_noise.constrain_fixed(1e-6, warning=False) #to avoid numerical problems
#         else:
#             self.model.obslayer.Gaussian_noise.constrain_bounded(1e-6,1e6, warning=False) #to avoid numerical problems


#     def updateModel(self, X_all, Y_all, X_new, Y_new):


#     def predict(self, X):
#         if X.ndim==1: X = X[None,:]
#         m, v = self.model.predict(X)
#         v = np.clip(v, 1e-10, np.inf)
#         return m, np.sqrt(v)

#     def get_fmin(self):

#         return 
    
#     def predict_withGradients(self, X):
#         if X.ndim==1: X = X[None,:]
#         m, v = self.model.predict(X)
#         v = np.clip(v, 1e-10, np.inf)
#         dmdx, dvdx = self.model.predictive_gradients(X)
#         dmdx = dmdx[:,:,0]
#         dsdx = dvdx / (2*np.sqrt(v))
#         return m, np.sqrt(v), dmdx, dsdx
    

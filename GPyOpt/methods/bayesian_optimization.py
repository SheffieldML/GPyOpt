# # Copyright (c) 2015, Javier Gonzalez
# # Copyright (c) 2015, the GPy Authors (see GPy AUTHORS.txt)
# # Licensed under the BSD 3-clause license (see LICENSE.txt)

# import GPy
# import deepgp
# import numpy as np
# import time
# from ..core.acquisition import AcquisitionEI, AcquisitionMPI, AcquisitionLCB, AcquisitionEL
# from ..core.bo import BO
# from ..util.general import samples_multidimensional_uniform, reshape, evaluate_function
# from ..util.stats import initial_design
# import warnings
# warnings.filterwarnings("ignore")


# class BayesianOptimization(BO):

# # new interface
#     def __init__(self, f, bounds=None, model_type=None, X=None, Y=None, initial_design_numdata = None, 
#         initial_design_type='random', model_optimize_interval=1, acquisition_type ='EI', acquisition_par = 0.00, 
#         model_optimize_restarts=3, normalize=False, exact_feval=False, verbosity=0):


# # old interface
#     def __init__(self, f, bounds=None, kernel=None, X=None, Y=None, numdata_initial_design = None, 
#         type_initial_design='random', model_optimize_interval=1, acquisition='EI',
#         acquisition_par= 0.00, model_optimize_restarts=10, sparseGP=False, num_inducing=None, normalize=False,
#         exact_feval=False, verbosity=0):




#         # --- Initialize internal parameters
#         self.input_dim = len(bounds)
#         self.normalize = normalize
#         self.exact_feval = exact_feval
#         self.model_optimize_interval = model_optimize_interval
#         self.model_optimize_restarts = model_optimize_restarts
#         self.verbosity = verbosity
#         self.first_time_optimization = True  
        
#          # --- Initialize objective function
#         if f==None: 
#             print 'Function to optimize is required.'
#         else:
#             self.f = f

#         # --- Initialize bounds
#         if bounds==None:
#             raise 'Box constraints are needed. Please insert box constrains.'
#         else:
#             self.bounds = bounds

#         # --- Initialize design
#         self._init_design(X,Y,initial_design_type,initial_design_numdata)

#         # --- Initialize model
#         self._init_model(model_type)
        
#         # --- Initialize acquisition
#         self._init_acquisition(acquisition_type,acquisition_par)



#     def _init_design(self, X, Y, initial_design_type, initial_design_numdata):

#         self.initial_design_type = initial_design_type

#         if  initial_design_numdata==None:
#             self.initial_design_numdata = 3*self.input_dim
#         else:
#             self.initial_design_numdata = initial_design_numdata

#         # Case 1:
#         if X==None:
#             if Y!=None:
#                 warnings.warn("User supplied initial Y without matching X")
#             self.X = initial_design(self.initial_design_type, self.bounds, self.initial_design_numdata)
#             self.Y, self.Y_cost = evaluate_function(self.f,self.X)

#         # Case 2
#         elif Y==None:
#             self.X = X
#             self.Y, self.Y_cost = evaluate_function(f,self.X)

#         # Case 3
#         else:
#             self.X = X
#             self.Y = Y

#     def _init_acquisition(self, acquisition_type, acquisition_par):
#         self.acqu_name = acquisition_type
#         if  acquisition_par == None:
#             self.acquisition_par = 0
#         else:
#             self.acquisition_par = acquisition_par

#         if acquisition_type==None or acquisition_type=='EI':
#             acq = AcquisitionEI(acquisition_par)
#         elif acquisition_type=='MPI':
#             acq = AcquisitionMPI(acquisition_par)
#         elif acquisition_type=='LCB':
#             acq = AcquisitionLCB(acquisition_par)
#         elif acquisition_type=='EL':
#             acq = AcquisitionEL(acquisition_par)
#         else:
#             print 'The selected acquisition function is not valid.'
        
#         if (acquisition_type=='EI' or acquisition_type=='MPI' or acquisition_type =='LCB' or acquisition_type =='EL' ):
#             super(BayesianOptimization ,self).__init__(acquisition_func=acq)


#     def _init_model(self,model_type):
#         '''
#         Initializes the model over *f*. The parameters can be changed after.
#         :param X: input observations.
#         :param Y: output values.
#         '''
#         if model_type == None:
#             self.model_type = 'gp'
#         else: 
#             self.model_type = model_type

#         # --- Other models can be added 
#         if self.model_type == 'sparsegp' or self.model_type == 'gp':
#             self._init_gp()

#         elif self.model_type == 'deepgp' or  self.model_type == 'deepgp_back_constraint':
#             self._init_deepgp()

       

    

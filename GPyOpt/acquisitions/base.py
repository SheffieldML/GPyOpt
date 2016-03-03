# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

class AcquisitionBase(object):
    """
    Base class for acquisition functions in Bayesian Optimization
    
    TODO for models that do not have gradients
    """

    def __init__(self, model, space, optimizer):
        self.model = model
        self.space = space
        self.optimizer = optimizer
        self.analytical_gradient_acq = self.model.analytical_gradient_prediction # flag from the model to test if gradients are available

    def acquisition_function(self, x):
        pass

    def acquisition_function_withGradients(self, x):
        pass
    
    def optimize(self):
        if not self.analytical_gradient_acq:
            out = self.optimizer.optimize(f=self.acquisition_function)[0]
        else:
            out = self.optimizer.optimize(f=self.acquisition_function, f_df=self.acquisition_function_withGradients)[0]
        return out
    
    
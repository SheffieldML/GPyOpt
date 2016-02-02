
class AcquisitionBase(object):
    """
    Base class for acquisition functions in Bayesian Optimization
    """
    def __init__(self, model, space, optimizer):
        self.model = model
        self.space = space
        self.optimizer = optimizer

    def acquisition_function(self, x):
        pass

    def acquisition_function_withGradients(self, x):
        pass
    
    def optimize(self):
        return self.optimizer.optimize(f=self.acquisition_function, f_df=self.acquisition_function_withGradients)[0]
    
    
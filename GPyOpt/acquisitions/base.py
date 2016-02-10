
class AcquisitionBase(object):
    """
    Base class for acquisition functions in Bayesian Optimization
    
    TODO for models that do not have gradients
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
        if not hasattr(self.model, 'predict_withGradients'):
            out = self.optimizer.optimize(f=self.acquisition_function, f_df=None)[0]
        else:
            out = self.optimizer.optimize(f=self.acquisition_function, f_df=self.acquisition_function_withGradients)[0]
        return out
    
    
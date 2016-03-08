from .base import EvaluatorBase

class Sequential(EvaluatorBase):
    """
    Class for Expected improvement acquisition functions.
    """
    def __init__(self, acquisition, batch_size=1):
        super(Sequential, self).__init__(acquisition, batch_size)
        #self.acquisition = acquisition

    def compute_batch(self):
    	return self.acquisition.optimize()



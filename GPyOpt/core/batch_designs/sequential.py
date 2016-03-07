from .base import BatchMethodBase

class Sequential(BatchMethodBase):
    """
    Class for Expected improvement acquisition functions.
    """
    def __init__(self, acquisition, batch_size):
        super(Sequential, self).__init__(acquisition, batch_size)
        self.acquisition = acquisition
        self.batch_size = batch_size

    def compute_batch(self):
    	return self.acquisition.optimize()



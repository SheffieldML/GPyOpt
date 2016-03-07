

class BatchMethodBase(object):
    """
    Base class for the batch method to use
    
    TODO for models that do not have gradients
    """

    def __init__(self, acquisition, batch_size, **kwargs):
    	self.acquisition = acquisition
    	self.batch_size = batch_size

    def compute_batch(self):
    	pass









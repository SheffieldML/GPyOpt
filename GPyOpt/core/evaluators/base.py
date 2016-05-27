# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

class EvaluatorBase(object):
    """
    Base class for the batch method to use
    
    TODO for models that do not have gradients
    """

    def __init__(self, acquisition, batch_size, **kwargs):
    	self.acquisition = acquisition
    	self.batch_size = batch_size

    def compute_batch(self):
    	pass









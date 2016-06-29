# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from .base import EvaluatorBase
from ...util.stats import initial_design
import numpy as np

class RandomBatch(EvaluatorBase):
    """
    Class for a random batch method. The first element of the batch is selected by optimizing the acquisition in a standard way. The remaining elements are
    selected uniformly random in the domain of the objective.

    :param acquisition: acquisition function to be used to compute the batch.
    :param batch size: the number of elements in the batch.
    :normalize_Y: whether to normalize the outputs.

    """
    def __init__(self, acquisition, batch_size):
        super(RandomBatch, self).__init__(acquisition, batch_size)
        self.acquisition = acquisition
        self.batch_size = batch_size

    def compute_batch(self):
        """
        Adds to the first location batch_size-1 randomly selected elements.
        """
        return np.vstack((self.acquisition.optimize(),initial_design('random',self.acquisition.space,self.batch_size-1)))  

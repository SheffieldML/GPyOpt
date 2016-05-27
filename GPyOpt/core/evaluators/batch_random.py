# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from .base import EvaluatorBase
from ...util.stats import initial_design
import numpy as np

class RandomBatch(EvaluatorBase):
    """
    Class for Expected improvement acquisition functions.
    """
    def __init__(self, acquisition, batch_size):
        super(RandomBatch, self).__init__(acquisition, batch_size)
        self.acquisition = acquisition
        self.batch_size = batch_size

    def compute_batch(self):
        return np.vstack((self.acquisition.optimize(),initial_design('random',self.acquisition.space,self.batch_size-1)))  

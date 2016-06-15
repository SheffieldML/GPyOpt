# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

class EvaluatorBase(object):
    """
    Base class for the evaluator of the function. This class handles both sequential and batch evaluators.
    
    """

    def __init__(self, acquisition, batch_size, **kwargs):
        self.acquisition = acquisition
        self.batch_size = batch_size

    def compute_batch(self):
        pass









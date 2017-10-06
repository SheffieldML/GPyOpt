# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from .base import EvaluatorBase


class Sequential(EvaluatorBase):
    """
    Class for standard Sequential Bayesian optimization methods.

    :param acquisition: acquisition function to be used to compute the batch.
    :param batch size: it is 1 by default since this class is only used for sequential methods.
    """

    def __init__(self, acquisition, batch_size=1):
        super(Sequential, self).__init__(acquisition, batch_size)

    def compute_batch(self, duplicate_manager=None,context_manager=None):
        """
        Selects the new location to evaluate the objective.
        """
        x, _ = self.acquisition.optimize(duplicate_manager=duplicate_manager)
        return x

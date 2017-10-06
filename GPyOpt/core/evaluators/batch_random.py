# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from .base import SamplingBasedBatchEvaluator

from ...optimization.anchor_points_generator import RandomAnchorPointsGenerator
import numpy as np

class RandomBatch(SamplingBasedBatchEvaluator):
    """
    Class for a random batch method. The first element of the batch is selected by optimizing the acquisition in a standard way. The remaining elements are
    selected uniformly random in the domain of the objective.

    :param acquisition: acquisition function to be used to compute the batch.
    :param batch size: the number of elements in the batch.

    """
    def __init__(self, acquisition, batch_size):

        super(RandomBatch, self).__init__(acquisition, batch_size)

    def initialize_batch(self, duplicate_manager=None,context_manager=None):

        x, _ = self.acquisition.optimize(duplicate_manager=duplicate_manager)

        return x

    def get_anchor_points(self, duplicate_manager=None,context_manager=None):

        design_type, unique = "random", False
        if duplicate_manager:
            unique = True

        anchor_points_generator = RandomAnchorPointsGenerator(self.space, design_type)
        return anchor_points_generator.get(num_anchor=self.num_anchor, duplicate_manager=duplicate_manager, unique=unique, context_manager = self.acquisition.optimizer.context_manager)

    def optimize_anchor_point(self, a, duplicate_manager=None,context_manager=None):

        return a

    def compute_batch_without_duplicate_logic(self, context_manager=None):

        x, anchor_points = self.initialize_batch(), self.get_anchor_points()

        return np.vstack((x, anchor_points[:(self.batch_size - 1), :]))

# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from .base import SamplingBasedBatchEvaluator
from ...optimization.anchor_points_generator import ThompsonSamplingAnchorPointsGenerator
from ...optimization.optimizer import OptLbfgs, apply_optimizer, choose_optimizer
import numpy as np

class ThompsonBatch(SamplingBasedBatchEvaluator):
    """
    Class for a Thompson batch method. Elements are selected iteratively using the current acquistion function but exploring the models
    by using Thompson sampling

    :param acquisition: acquisition function to be used to compute the batch.
    :param batch size: the number of elements in the batch.

    """
    def __init__(self, acquisition, batch_size):

        super(ThompsonBatch, self).__init__(acquisition, batch_size)
        self.model              = self.acquisition.model
        self.optimizer_name     = 'lbfgs'
        self.f                  = self.acquisition.acquisition_function
        self.f_df               = self.acquisition.acquisition_function_withGradients
        self.space              = self.acquisition.space

    def initialize_batch(self, duplicate_manager=None, context_manager=None):

        return None

    def get_anchor_points(self, duplicate_manager=None, context_manager=None):
        design_type, unique = "random", False
        if duplicate_manager:
            unique = True

        anchor_points_generator = ThompsonSamplingAnchorPointsGenerator(self.space, design_type, model=self.model)
        return anchor_points_generator.get(num_anchor=self.num_anchor, duplicate_manager=duplicate_manager, unique=unique, context_manager = self.context_manager)

    def optimize_anchor_point(self, a, duplicate_manager=None, context_manager=None):
        ### --- Update the bounds of the default optimizer according to the context_manager
        if context_manager:
            bounds = self.context_manager.noncontext_bounds
        else:
            bounds = self.space.get_bounds()

        self.local_optimizer = choose_optimizer(self.optimizer_name, bounds)

        ### --- Run the local optimizer
        x, _ = apply_optimizer(self.local_optimizer, a, f=self.f, df=None, f_df=self.f_df, duplicate_manager=duplicate_manager, context_manager = self.context_manager, space=self.space)
        return self.space.round_optimum(x)

    def compute_batch_without_duplicate_logic(self, context_manager=None):
        anchor_points = self.get_anchor_points(context_manager=context_manager)
        return np.vstack([self.optimize_anchor_point(a, context_manager=context_manager) for a, _ in zip(anchor_points, range(self.batch_size))])

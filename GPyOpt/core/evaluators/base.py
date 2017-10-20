# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np


class EvaluatorBase(object):
    """
    Base class for the evaluator of the function. This class handles both sequential and batch evaluators.

    """

    def __init__(self, acquisition, batch_size, **kwargs):
        self.acquisition = acquisition
        self.batch_size = batch_size

    def compute_batch(self, duplicate_manager=None, context_manager=None):
        raise NotImplementedError("Need to implement compute_batch.")


class SamplingBasedBatchEvaluator(EvaluatorBase):
    """
    This class handles specific types of batch evaluators, based on the sampling of anchor points (examples are random and Thompson sampling).

    """

    def __init__(self, acquisition, batch_size, **kwargs):
        self.acquisition = acquisition
        self.batch_size = batch_size
        self.space = acquisition.space
        # The following number of anchor points is heuristically picked, to obtain good and various batches
        self.num_anchor = 5*batch_size

    def initialize_batch(self, duplicate_manager=None, context_manager=None):
        raise NotImplementedError("Need to implement initialize_batch.")

    def get_anchor_points(self, duplicate_manager=None, context_manager=None):
        raise NotImplementedError("Need to implement get_anchor_points.")

    def optimize_anchor_point(self, a, duplicate_manager=None, context_manager=None):
        raise NotImplementedError("Need to implement optimize_anchor_point.")

    def compute_batch_without_duplicate_logic(self,context_manager=None):
        raise NotImplementedError("Need to implement compute_batch_without_duplicate_logic.")

    def compute_batch(self, duplicate_manager=None, context_manager=None):

        self.context_manager = context_manager

        # Easy case where we do not care about having duplicates suggested
        if not duplicate_manager:
            return self.compute_batch_without_duplicate_logic(context_manager=self.context_manager)

        batch, already_suggested_points = [], duplicate_manager.unique_points.copy()

        anchor_points = self.get_anchor_points(duplicate_manager=duplicate_manager, context_manager=self.context_manager)

        x0 = self.initialize_batch(duplicate_manager=duplicate_manager, context_manager = self.context_manager)

        if np.any(x0):
            batch.append(x0)
            already_suggested_points.add(self.zip_and_tuple(x0))

        for a in anchor_points:
            x = self.optimize_anchor_point(a, duplicate_manager=duplicate_manager, context_manager = self.context_manager)

            # We first try to add the optimized anchor point; if we cannot, we then try the initial anchor point.
            zipped_x = self.zip_and_tuple(x)

            if zipped_x not in already_suggested_points:
                batch.append(x)
                already_suggested_points.add(zipped_x)
            else:
                zipped_a = self.zip_and_tuple(a)

                if zipped_a not in already_suggested_points:
                    batch.append(a)
                    already_suggested_points.add(zipped_a)

            if len(batch) == self.batch_size:
                break

        if len(batch) < self.batch_size:
            # Note that the case where anchor_points is empty is handled in self.get_anchor_points that would throw a FullyExploredOptimizationDomainError
            print("Warning: the batch of requested size {} could not be entirely filled in (only {} points)".format(self.batch_size, len(batch)))

        return np.vstack(batch)

    def zip_and_tuple(self, x):
        """
        convenient helper
        :param x: input configuration in the model space
        :return: zipped x as a tuple
        """
        return tuple(self.space.zip_inputs(np.atleast_2d(x)).flatten())

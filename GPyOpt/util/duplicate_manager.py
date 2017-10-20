# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from ..core.task.space import Design_space

class DuplicateManager(object):
    """
    Class to manage potential duplicates in the suggested candidates.

    :param space: object managing all the logic related the domain of the optimization
    :param zipped_X: matrix of evaluated configurations
    :param pending_zipped_X: matrix of configurations in the pending state
    :param ignored_zipped_X: matrix of configurations that the user desires to ignore (e.g., because they may have led to failures)
    """

    def __init__(self, space, zipped_X, pending_zipped_X=None, ignored_zipped_X=None):

        self.space = space

        self.unique_points = set()
        self.unique_points.update(tuple(x.flatten()) for x in zipped_X)

        if np.any(pending_zipped_X):
            self.unique_points.update(tuple(x.flatten()) for x in pending_zipped_X)

        if np.any(ignored_zipped_X):
            self.unique_points.update(tuple(x.flatten()) for x in ignored_zipped_X)


    def is_zipped_x_duplicate(self, zipped_x):
        """
        param: zipped_x: configuration assumed to be zipped
        """
        return tuple(zipped_x.flatten()) in self.unique_points

    def is_unzipped_x_duplicate(self, unzipped_x):
        """
        param: unzipped_x: configuration assumed to be unzipped
        """
        return self.is_zipped_x_duplicate(self.space.zip_inputs(np.atleast_2d(unzipped_x)))

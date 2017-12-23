import numpy as np

from ..core.errors import InvalidConfigError

from .base import ExperimentDesign
from .random_design import RandomDesign

class LatinDesign(ExperimentDesign):
    """
    Latin experiment design.
    Uses random design for non-continuous variables, and latin hypercube for continuous ones
    """
    def __init__(self, space):
        if space.has_constraints():
            raise InvalidConfigError('Sampling with constraints is not allowed by latin design')
        super(LatinDesign, self).__init__(space)

    def get_samples(self, init_points_count):
        samples = np.empty((init_points_count, self.space.dimensionality))

        # Use random design to fill non-continuous variables
        random_design = RandomDesign(self.space)
        random_design.fill_noncontinous_variables(samples)

        if self.space.has_continuous():
            bounds = self.space.get_continuous_bounds()
            lower_bound = np.asarray(bounds)[:,0].reshape(1, len(bounds))
            upper_bound = np.asarray(bounds)[:,1].reshape(1, len(bounds))
            diff = upper_bound - lower_bound

            from pyDOE import lhs
            X_design_aux = lhs(len(self.space.get_continuous_bounds()), init_points_count, criterion='center')
            I = np.ones((X_design_aux.shape[0], 1))
            X_design = np.dot(I, lower_bound) + X_design_aux * np.dot(I, diff)

            samples[:, self.space.get_continuous_dims()] = X_design

        return samples
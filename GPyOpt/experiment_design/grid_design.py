import numpy as np

from ..core.errors import InvalidConfigError

from .base import ExperimentDesign
from .random_design import RandomDesign

class GridDesign(ExperimentDesign):
    """
    Grid experiment design.
    Uses random design for non-continuous variables, and square grid for continuous ones
    """

    def __init__(self, space):
        if space.has_constraints():
            raise InvalidConfigError('Sampling with constraints is not allowed by grid design')
        super(GridDesign, self).__init__(space)

    def _adjust_init_points_count(self, init_points_count):
        # TODO: log this
        print('Note: in grid designs the total number of generated points is the smallest closest integer of n^d to the selected amount of points')
        continuous_dims = len(self.space.get_continuous_dims())
        self.data_per_dimension = iroot(continuous_dims, init_points_count)
        return self.data_per_dimension**continuous_dims

    def get_samples(self, init_points_count):
        """
        This method may return less points than requested.
        The total number of generated points is the smallest closest integer of n^d to the selected amount of points.
        """

        init_points_count = self._adjust_init_points_count(init_points_count)
        samples = np.empty((init_points_count, self.space.dimensionality))

        # Use random design to fill non-continuous variables
        random_design = RandomDesign(self.space)
        random_design.fill_noncontinous_variables(samples)

        if self.space.has_continuous():
            X_design = multigrid(self.space.get_continuous_bounds(), self.data_per_dimension)
            samples[:,self.space.get_continuous_dims()] = X_design

        return samples

# Computes integer root
# The greatest integer whose k-th power is less than or equal to n
# That is the greatest x such that x^k <= n
def iroot(k, n):
    # Implements Newton Iroot algorithm
    # Details can be found here: https://www.akalin.com/computing-iroot
    # In a nutshell, it constructs a decreasing number series
    # that is guaranteed to terminate at the required integer root
    u, s = n, n+1
    while u < s:
        s = u
        t = (k-1) * s + n // pow(s, k-1)
        u = t // k
    return s

def multigrid(bounds, points_count):
    """
    Generates a multidimensional lattice
    :param bounds: box constraints
    :param points_count: number of points per dimension.
    """
    if len(bounds)==1:
        return np.linspace(bounds[0][0], bounds[0][1], points_count).reshape(points_count, 1)
    x_grid_rows = np.meshgrid(*[np.linspace(b[0], b[1], points_count) for b in bounds])
    x_grid_columns = np.vstack([x.flatten(order='F') for x in x_grid_rows]).T
    return x_grid_columns
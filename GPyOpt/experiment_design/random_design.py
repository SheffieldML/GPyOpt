import numpy as np

from .base import ExperimentDesign
from ..core.task.variables import BanditVariable, DiscreteVariable, CategoricalVariable


class RandomDesign(ExperimentDesign):
    """
    Random experiment design.
    Random values for all variables within the given bounds.
    """
    def __init__(self, space):
        super(RandomDesign, self).__init__(space)

    def get_samples(self, init_points_count):
        if self.space.has_constraints():
            return self.get_samples_with_constraints(init_points_count)
        else:
            return self.get_samples_without_constraints(init_points_count)

    def get_samples_with_constraints(self, init_points_count):
        """
        Draw random samples and only save those that satisfy constraints
        Finish when required number of samples is generated
        """
        samples = np.empty((0, self.space.dimensionality))

        while samples.shape[0] < init_points_count:
            domain_samples = self.get_samples_without_constraints(init_points_count)
            valid_indices = (self.space.indicator_constraints(domain_samples) == 1).flatten()
            if sum(valid_indices) > 0:
                valid_samples = domain_samples[valid_indices,:]
                samples = np.vstack((samples,valid_samples))

        return samples[0:init_points_count,:]

    def fill_noncontinous_variables(self, samples):
        """
        Fill sample values to non-continuous variables in place
        """
        init_points_count = samples.shape[0]

        for (idx, var) in enumerate(self.space.space_expanded):
            if isinstance(var, DiscreteVariable) or isinstance(var, CategoricalVariable) :
                sample_var = np.atleast_2d(np.random.choice(var.domain, init_points_count))
                samples[:,idx] = sample_var.flatten()

            # sample in the case of bandit variables
            elif isinstance(var, BanditVariable):
                # Bandit variable is represented by a several adjacent columns in the samples array
                idx_samples = np.random.randint(var.domain.shape[0], size=init_points_count)
                bandit_idx = np.arange(idx, idx + var.domain.shape[1])
                samples[:, bandit_idx] = var.domain[idx_samples,:]


    def get_samples_without_constraints(self, init_points_count):
        samples = np.empty((init_points_count, self.space.dimensionality))

        self.fill_noncontinous_variables(samples)

        if self.space.has_continuous():
            X_design = samples_multidimensional_uniform(self.space.get_continuous_bounds(), init_points_count)
            samples[:, self.space.get_continuous_dims()] = X_design

        return samples

def samples_multidimensional_uniform(bounds, points_count):
    """
    Generates a multidimensional grid uniformly distributed.
    :param bounds: tuple defining the box constraints.
    :points_count: number of data points to generate.
    """
    dim = len(bounds)
    Z_rand = np.zeros(shape=(points_count, dim))
    for k in range(0,dim):
        Z_rand[:,k] = np.random.uniform(low=bounds[k][0], high=bounds[k][1], size=points_count)
    return Z_rand
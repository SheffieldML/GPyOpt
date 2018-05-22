import emcee

from ..experiment_design import initial_design

class McmcSampler(object):
    def __init__(self, space):
        """
        Creates an instance of the sampler.

        Parameters:
            space - variable space
        """
        self.space = space

    def get_samples(self, n_samples, log_p_function, burn_in_steps=50):
        """
        Generates samples.

        Parameters:
            n_samples - number of samples to generate
            log_p_function - a function that returns log density for a specific sample
            burn_in_steps - number of burn-in steps for sampling

        Returns a tuple of two lists: (samples, log_p_function values for samples)
        """

        raise NotImplementedError

class AffineInvariantEnsembleSampler(McmcSampler):
    def __init__(self, space):
        """
        Creates an instance of the affine invariant ensemble sampler.

        Parameters:
            space - variable space
        """
        super(AffineInvariantEnsembleSampler, self).__init__(space)

    def get_samples(self, n_samples, log_p_function, burn_in_steps=50):
        """
        Generates samples.

        Parameters:
            n_samples - number of samples to generate
            log_p_function - a function that returns log density for a specific sample
            burn_in_steps - number of burn-in steps for sampling

        Returns a tuple of two array: (samples, log_p_function values for samples)
        """
        restarts = initial_design('random', self.space, n_samples)
        sampler = emcee.EnsembleSampler(n_samples, self.space.input_dim(), log_p_function)
        samples, samples_log, _ = sampler.run_mcmc(restarts, burn_in_steps)

        # make sure we have an array of shape (n samples, space input dim)
        if len(samples.shape) == 1:
            samples = samples.reshape(-1, 1)
        samples_log = samples_log.reshape(-1, 1)

        return samples, samples_log
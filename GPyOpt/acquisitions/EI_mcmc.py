# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from .EI import AcquisitionEI

class AcquisitionEI_MCMC(AcquisitionEI):
    """
    Integrated Expected improvement acquisition function

    :param model: GPyOpt class of model
    :param space: GPyOpt class of domain
    :param optimizer: optimizer of the acquisition. Should be a GPyOpt optimizer
    :param cost_withGradients: function
    :param jitter: positive value to make the acquisition more explorative

    .. Note:: allows to compute the Improvement per unit of cost

    """


    def __init__(self, model, space, optimizer=None, cost_withGradients=None, jitter=0.01):
        super(AcquisitionEI_MCMC, self).__init__(model, space, optimizer)
        
        assert self.model.MCMC_sampler, 'Samples from the hyperparameters are needed to compute the integrated EI'

    def acquisition_function(self,x):
        """
        Integrated Expected Improvement
        """    
        means, stds = self.model.predict(x)
        fmin = self.model.get_fmin()
        f_acqu = 0
        for m,s in zip(means, stds):
            f_acqu += self._compute_acq(m, s, fmin, x)
        return f_acqu/(len(means))

    def acquisition_function_withGradients(self, x):
        """
        Integrated Expected Improvement and its derivative
        """
        means, stds, dmdxs, dsdxs = self.model.predict_withGradients(x)
        fmin = self.model.get_fmin()
        f_acqu = None
        df_acqu = None
        for m, s, dmdx, dsdx in zip(means, stds, dmdxs, dsdxs):
            if f_acqu is None:
                f_acqu, df_acqu = self._compute_acq_withGradients(m, s, fmin, dmdx, dsdx, x)
            else:
                f, df = self._compute_acq_withGradients(m, s, fmin, dmdx, dsdx, x)
                f_acqu += f
                df_acqu += df
        return f_acqu/(len(means)), df_acqu/(len(means))
        









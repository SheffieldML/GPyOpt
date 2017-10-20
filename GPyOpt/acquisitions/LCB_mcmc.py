# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from .LCB import AcquisitionLCB

class AcquisitionLCB_MCMC(AcquisitionLCB):
    """
    Integrated GP-Lower Confidence Bound acquisition function

    :param model: GPyOpt class of model
    :param space: GPyOpt class of domain
    :param optimizer: optimizer of the acquisition. Should be a GPyOpt optimizer
    :param cost_withGradients: function
    :param exploration_weight: positive parameter to control exploration / exploitation

    .. Note:: allows to compute the Improvement per unit of cost

    """

    analytical_gradient_prediction = True
    
    def __init__(self, model, space, optimizer=None, cost_withGradients=None, exploration_weight=2):
        super(AcquisitionLCB_MCMC, self).__init__(model, space, optimizer,cost_withGradients,exploration_weight)
        assert self.model.MCMC_sampler, 'Samples from the hyper-parameters are needed to compute the integrated GP-LCB'

    def _compute_acq(self,x):
        """
        Integrated GP-Lower Confidence Bound
        """ 
        means, stds = self.model.predict(x)
        f_acqu = 0
        for m,s in zip(means, stds):
            f_acqu += -m + self.exploration_weight * s
        return f_acqu/(len(means))

    def _compute_acq_withGradients(self, x):
        """
        Integrated GP-Lower Confidence Bound and its derivative
        """
        means, stds, dmdxs, dsdxs = self.model.predict_withGradients(x)
        f_acqu = None
        df_acqu = None
        for m, s, dmdx, dsdx in zip(means, stds, dmdxs, dsdxs):  
            f = -m + self.exploration_weight * s        
            df = -dmdx + self.exploration_weight * dsdx
            if f_acqu is None:
                f_acqu = f        
                df_acqu = df
            else:
                f_acqu += f
                df_acqu += df
        return f_acqu/(len(means)), df_acqu/(len(means))




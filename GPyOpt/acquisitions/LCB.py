# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from .base import AcquisitionBase
from ..util.general import get_quantiles
from ..core.task.cost import constant_cost_withGradients

class AcquisitionLCB(AcquisitionBase):
    """
    GP-Lower Confidence Bound acquisition function

    :param model: GPyOpt class of model
    :param space: GPyOpt class of domain
    :param optimizer: optimizer of the acquisition. Should be a GPyOpt optimizer
    :param cost_withGradients: function
    :param exploration_weight: positive parameter to comtrol exploration/explotitation

    .. Note:: allows to compute the Improvement per unit of cost

    """
    
    analytical_gradient_prediction = True

    def __init__(self, model, space, optimizer=None, cost_withGradients=None, exploration_weight=2):
        self.optimizer = optimizer
        super(AcquisitionLCB, self).__init__(model, space, optimizer)
        self.exploration_weight = exploration_weight

        if cost_withGradients is not None:
            print('The set cost function is ignored! LBC acquisition does not make sense with cost.')  
    
    def _compute_acq(self, m, s, x):
        """
        Computes the GP-Lower Confidence Bound per unit of cost
        """
        f_acqu = -m + self.exploration_weight * s
        cost_x, _ = self.cost_withGradients(x)
        return -(f_acqu*self.space.indicator_constrains(x))/cost_x

    def acquisition_function(self,x):
        """
        GP-Lower Confidence Bound
        """
        m, s = self.model.predict(x)
        return self._compute_acq(m, s, x) # note: returns negative value for posterior minimization

    def acquisition_function_withGradients(self, x):
        """
        Computes the GP-Lower Confidence Bound and its derivative (has a very easy derivative!)
        """
        m, s, dmdx, dsdx = self.model.predict_withGradients(x)
        return self._compute_acq_withGradients(m, s, dmdx, dsdx, x)

    def _compute_acq_withGradients(self, m, s, dmdx, dsdx, x):
        """
        GP-Lower Confidence Bound and its derivative
        """
        f_acqu = -m + self.exploration_weight * s
        df_acqu = -dmdx + self.exploration_weight * dsdx
        cost_x, cost_grad_x = self.cost_withGradients(x)
        f_acq_cost = f_acqu/cost_x
        df_acq_cost = (df_acqu*cost_x - f_acqu*cost_grad_x)/(cost_x**2)
        return -f_acq_cost*self.space.indicator_constrains(x), -df_acq_cost*self.space.indicator_constrains(x)

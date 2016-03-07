# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from .base import AcquisitionBase
from ..util.general import get_quantiles
from ..core.task.cost import constant_cost_withGradients

class AcquisitionLCB(AcquisitionBase):
    """
    Class for Expected improvement acquisition functions.
    """
    def __init__(self, model, space, optimizer=None, cost_withGradients=None, exploration_weight=2):
        optimizer = optimizer
        super(AcquisitionLCB, self).__init__(model, space, optimizer)
        self.exploration_weight = exploration_weight

        if cost_withGradients == None:
            self.cost_withGradients = constant_cost_withGradients
        else:
            print 'LBC acquisition does now make sense with cost. Cost set to constant.'  
            self.cost_withGradients = constant_cost_withGradients
    
    def _compute_acq(self, m, s, x):
        f_acqu = -m + self.exploration_weight * s
        cost_x, _ = self.cost_withGradients(x)
        return -(f_acqu*self.space.indicator_constrains(x))/cost_x

    def acquisition_function(self,x):
        m, s = self.model.predict(x)
        return self._compute_acq(m, s, x) # note: returns negative value for posterior minimization

    def acquisition_function_withGradients(self, x):
        m, s, dmdx, dsdx = self.model.predict_withGradients(x)
        return self._compute_acq_withGradients(m, s, dmdx, dsdx, x)

    def _compute_acq_withGradients(self, m, s, dmdx, dsdx, x):
        f_acqu = -m + self.exploration_weight * s
        df_acqu = -dmdx + self.exploration_weight * dsdx
        cost_x, cost_grad_x = self.cost_withGradients(x)
        f_acq_cost = f_acqu/cost_x
        df_acq_cost = (df_acqu*cost_x - f_acqu*cost_grad_x)/(cost_x**2)
        return -f_acq_cost*self.space.indicator_constrains(x), -df_acq_cost*self.space.indicator_constrains(x)

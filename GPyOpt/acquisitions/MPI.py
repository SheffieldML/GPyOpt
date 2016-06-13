# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from .base import AcquisitionBase
from ..util.general import get_quantiles
from ..core.task.cost import constant_cost_withGradients

class AcquisitionMPI(AcquisitionBase):
    """
    Maximum probability of improvement acquisition function

    :param model: GPyOpt class of model
    :param space: GPyOpt class of domain
    :param optimizer: optimizer of the acquisition. Should be a GPyOpt optimizer
    :param cost_withGradients: function
    :param jitter: positive value to make the acquisition more explorative

    .. Note:: allows to compute the Improvement per unit of cost

    """

    analytical_gradient_prediction = True

    def __init__(self, model, space, optimizer=None, cost_withGradients=None, jitter=0.01):
        self.optimizer = optimizer
        super(AcquisitionMPI, self).__init__(model, space, optimizer)
        self.jitter = jitter

        if cost_withGradients == None:
            self.cost_withGradients = constant_cost_withGradients
        else:
            self.cost_withGradients = cost_withGradients

    
    def _compute_acq(self, m, s, fmin, x):
        """
        Computes the Maximum Probability of Improvement per unit of cost
        """
        _, Phi,_ = get_quantiles(self.jitter, fmin, m, s)    
        f_acqu =  Phi
        cost_x, _ = self.cost_withGradients(x)
        return -(f_acqu*self.space.indicator_constrains(x))/cost_x

    def acquisition_function(self,x):
        """
        Maximum Probability of Improvement
        """
        m, s = self.model.predict(x)
        fmin = self.model.get_fmin()   
        return self._compute_acq(m, s, fmin, x)  # note: returns negative value for posterior minimization 

    def _compute_acq_withGradients(self, m, s, fmin, dmdx, dsdx, x):
        """
        Computes the Maximum Probability of Improvement and its derivative (has a very easy derivative!)
        """
        phi, Phi, u = get_quantiles(self.jitter, fmin, m, s)    
        f_acqu =  Phi        
        df_acqu = -(phi/s)* (dmdx + dsdx * u)
        cost_x, cost_grad_x = self.cost_withGradients(x)
        f_acq_cost = f_acqu/cost_x
        df_acq_cost = (df_acqu*cost_x - f_acqu*cost_grad_x)/(cost_x**2)
        return -f_acq_cost*self.space.indicator_constrains(x), -df_acq_cost*self.space.indicator_constrains(x)

    def acquisition_function_withGradients(self, x):
        """
        Maximum Probability of Improvement and its derivative
        """
        m, s, dmdx, dsdx = self.model.predict_withGradients(x)
        fmin = self.model.get_fmin()
        return self._compute_acq_withGradients(m, s, fmin, dmdx, dsdx, x)
    
    
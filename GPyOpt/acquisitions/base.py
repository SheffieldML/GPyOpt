# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from ..core.task.cost import constant_cost_withGradients

class AcquisitionBase(object):
    """
    Base class for acquisition functions in Bayesian Optimization

    :param model: GPyOpt class of model
    :param space: GPyOpt class of domain
    :param optimizer: optimizer of the acquisition. Should be a GPyOpt optimizer

    """

    analytical_gradient_prediction = False

    def __init__(self, model, space, optimizer, cost_withGradients=None):
        self.model = model
        self.space = space
        self.optimizer = optimizer
        self.analytical_gradient_acq = self.analytical_gradient_prediction and self.model.analytical_gradient_prediction # flag from the model to test if gradients are available

        if cost_withGradients is  None:
            self.cost_withGradients = constant_cost_withGradients
        else:
            self.cost_withGradients = cost_withGradients

    @staticmethod
    def fromDict(model, space, optimizer, cost_withGradients, config):
        raise NotImplementedError()

    def acquisition_function(self,x):
        """
        Takes an acquisition and weights it so the domain and cost are taken into account.
        """
        f_acqu = self._compute_acq(x)
        cost_x, _ = self.cost_withGradients(x)
        return -(f_acqu*self.space.indicator_constraints(x))/cost_x


    def acquisition_function_withGradients(self, x):
        """
        Takes an acquisition and it gradient and weights it so the domain and cost are taken into account.
        """
        f_acqu,df_acqu = self._compute_acq_withGradients(x)
        cost_x, cost_grad_x = self.cost_withGradients(x)
        f_acq_cost = f_acqu/cost_x
        df_acq_cost = (df_acqu*cost_x - f_acqu*cost_grad_x)/(cost_x**2)
        return -f_acq_cost*self.space.indicator_constraints(x), -df_acq_cost*self.space.indicator_constraints(x)

    def optimize(self, duplicate_manager=None):
        """
        Optimizes the acquisition function (uses a flag from the model to use gradients or not).
        """
        if not self.analytical_gradient_acq:
            out = self.optimizer.optimize(f=self.acquisition_function, duplicate_manager=duplicate_manager)
        else:
            out = self.optimizer.optimize(f=self.acquisition_function, f_df=self.acquisition_function_withGradients, duplicate_manager=duplicate_manager)
        return out

    def _compute_acq(self,x):

        raise NotImplementedError('')

    def _compute_acq_withGradients(self, x):

        raise NotImplementedError('')

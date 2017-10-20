# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from ...models import GPModel
import numpy as np


class  CostModel(object):
    """
    Class to handle the cost of evaluating the function.

    param cost_withGradients: function that returns the cost of evaluating the function and its gradient. By default
    no cost is used. Options are:
        - cost_withGradients is some pre-defined cost function. Should return numpy array as outputs.
        - cost_withGradients = 'evaluation_time'.

    .. Note:: if cost_withGradients = 'evaluation time' the evaluation time of the function is used to model a GP whose
    mean is used as cost.

    """

    def __init__(self, cost_withGradients):
        super(CostModel, self).__init__()

        self.cost_type = cost_withGradients

        # --- Set-up evaluation cost
        if self.cost_type is  None:
            self.cost_withGradients = constant_cost_withGradients
            self.cost_type = 'Constant cost'

        elif self.cost_type == 'evaluation_time':
            self.cost_model = GPModel()                                 
            self.cost_withGradients  = self._cost_gp_withGradients
            self.num_updates = 0
        else:
            self.cost_withGradients  = cost_withGradients
            self.cost_type  = 'Used defined cost'


    def _cost_gp(self,x):
        """
        Predicts the time cost of evaluating the function at x.
        """
        m, _, _, _= self.cost_model.predict_withGradients(x)
        return np.exp(m)

    def _cost_gp_withGradients(self,x):
        """
        Predicts the time cost and its gradient of evaluating the function at x.
        """
        m, _, dmdx, _= self.cost_model.predict_withGradients(x)
        return np.exp(m), np.exp(m)*dmdx

    def update_cost_model(self, x, cost_x):
        """
        Updates the GP used to handle the cost.

        param x: input of the GP for the cost model.
        param x_cost: values of the time cost at the input locations.
        """

        if self.cost_type == 'evaluation_time':
            cost_evals = np.log(np.atleast_2d(np.asarray(cost_x)).T)

            if self.num_updates == 0:
                X_all = x
                costs_all = cost_evals
            else:
                X_all = np.vstack((self.cost_model.model.X,x))
                costs_all = np.vstack((self.cost_model.model.Y,cost_evals))

            self.num_updates += 1
            self.cost_model.updateModel(X_all, costs_all, None, None)

def constant_cost_withGradients(x):
    """
    Constant cost function used by default: cost=1, d_cost =0.
    """
    return np.ones(x.shape[0])[:,None], np.zeros(x.shape)

from .base import AcquisitionBase
from ..util.general import get_quantiles

class AcquisitionLCB(AcquisitionBase):
    """
    Class for Expected improvement acquisition functions.
    """
    def __init__(self, model, space, optimizer=None, cost = None, exploration_weight=2):
        optimizer = optimizer
        super(AcquisitionLCB, self).__init__(model, space, optimizer)
        self.exploration_weight = exploration_weight
        if cost == None: 
            self.cost = lambda x: 1
        else:
            self.cost = cost
    
    def _compute_acq(self, m, s, x):
        f_acqu = -m + self.exploration_weight * s
        return -(f_acqu*self.space.indicator_constrains(x))/self.cost(x)

    def acquisition_function(self,x):
        m, s = self.model.predict(x)
        return self._compute_acq(m, s, x) # note: returns negative value for posterior minimization

    def acquisition_function_withGradients(self, x):
        m, s, dmdx, dsdx = self.model.predict_withGradients(x)
        return self._compute_acq_withGradients(m, s, dmdx, dsdx, x)

    def _compute_acq_withGradients(self, m, s, dmdx, dsdx, x):
        f_acqu = -m + self.exploration_weight * s
        df_acqu = -dmdx + self.exploration_weight * dsdx
        return -(f_acqu*self.space.indicator_constrains(x))/self.cost(x), -(df_acqu*self.space.indicator_constrains(x))/self.cost(x)



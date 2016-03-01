from .base import AcquisitionBase
from ..util.general import get_quantiles, constant_cost_withGradients


class AcquisitionEI(AcquisitionBase):
    """
    Class for Expected improvement acquisition functions.
    """
    def __init__(self, model, space, optimizer=None, cost_withGradients=None, jitter=0.01):
        optimizer = optimizer
        super(AcquisitionEI, self).__init__(model, space, optimizer)
        self.jitter = jitter

        if cost_withGradients == None:
            self.cost_withGradients = constant_cost_withGradients

    def _compute_acq(self, m, s, fmin, x):
        phi, Phi, _ = get_quantiles(self.jitter, fmin, m, s)    
        f_acqu = (fmin - m + self.jitter) * Phi + s * phi
        cost_x, _ = self.cost_withGradients(x)
        return -(f_acqu*self.space.indicator_constrains(x))/cost_x

    def acquisition_function(self,x):
        """
        Expected Improvement
        """
        m, s = self.model.predict(x)
        fmin = self.model.get_fmin()
        return self._compute_acq(m, s, fmin, x)
    
    def _compute_acq_withGradients(self, m, s, fmin, dmdx, dsdx, x):
        phi, Phi, _ = get_quantiles(self.jitter, fmin, m, s)    
        f_acqu = (fmin - m + self.jitter) * Phi + s * phi        
        df_acqu = dsdx * phi - Phi * dmdx
        cost_x, cost_grad_x = self.cost_withGradients(x)
        
        # Value of the acquisition relative to the cost 
        f_acq_cost = f_acqu/cost_x
        df_acq_cost = (df_acqu*cost_x - f_acqu*cost_grad_x)/(cost_x**2)
        return -f_acq_cost*self.space.indicator_constrains(x), -df_acq_cost*self.space.indicator_constrains(x)


    def acquisition_function_withGradients(self, x):
        """
        Derivative of the Expected Improvement (has a very easy derivative!)
        """
        m, s, dmdx, dsdx = self.model.predict_withGradients(x)
        fmin = self.model.get_fmin()
        return self._compute_acq_withGradients(m, s, fmin, dmdx, dsdx, x)
    
    
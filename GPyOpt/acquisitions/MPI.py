from .base import AcquisitionBase
from ..util.general import get_quantiles

class AcquisitionMPI(AcquisitionBase):
    """
    Class for Maximum probability of improvement acquisition functions.
    """
    def __init__(self, model, space, optimizer=None, cost_withGradients=None, jitter=0.01):
        optimizer = optimizer
        super(AcquisitionMPI, self).__init__(model, space, optimizer)
        self.jitter = jitter

        if cost_withGradients == None:
            self.cost_withGradients = constant_cost_withGradients
    
    def _compute_acq(self, m, s, fmin, x):
        _, Phi,_ = get_quantiles(self.jitter, fmin, m, s)    
        f_acqu =  Phi
        cost_x, _ = self.cost_withGradients(x)
        return -(f_acqu*self.space.indicator_constrains(x))/cost_x

    def acquisition_function(self,x):
        m, s = self.model.predict(x)
        fmin = self.model.get_fmin()   
        return self._compute_acq(m, s, fmin, x)  # note: returns negative value for posterior minimization 

    def _compute_acq_withGradients(self, m, s, fmin, dmdx, dsdx, x):
        phi, Phi, u = get_quantiles(self.jitter, fmin, m, s)    
        f_acqu =  Phi        
        df_acqu = -(phi/s)* (dmdx + dsdx * u)
        cost_x, cost_grad_x = self.cost_withGradients(x)
        f_acq_cost = f_acqu/cost_x
        df_acq_cost = (df_acqu*cost_x - f_acqu*cost_grad_x)/(cost_x**2)
        return -f_acq_cost*self.space.indicator_constrains(x), -df_acq_cost*self.space.indicator_constrains(x)

    def acquisition_function_withGradients(self, x):
        """
        Derivative of Maximum Probability of Improvement
        """
        m, s, dmdx, dsdx = self.model.predict_withGradients(x)
        fmin = self.model.get_fmin()
        return self._compute_acq_withGradients(m, s, fmin, dmdx, dsdx, x)
    
    
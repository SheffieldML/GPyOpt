from .base import AcquisitionBase
from ..util.general import get_quantiles

class AcquisitionMPI(AcquisitionBase):
    """
    Class for Expected improvement acquisition functions.
    """
    def __init__(self, model, space, optimizer=None, cost = None, jitter=0.01):
        optimizer = optimizer
        super(AcquisitionMPI, self).__init__(model, space, optimizer)
        self.jitter = jitter
        if cost == None: 
            self.cost = lambda x: 1
        else:
            self.cost = cost
    
    def acquisition_function(self,x):
        """
        Expected Improvement
        """
        m, s = self.model.predict(x)
        fmin = self.model.get_fmin()   
        _, Phi,_ = get_quantiles(self.jitter, fmin, m, s)    
        f_acqu =  Phi
        return -f_acqu  # note: returns negative value for posterior minimization 

    def acquisition_function_withGradients(self, x):
        """
        Derivative of Maximum Probability of Improvement
        """
        m, s, dmdx, dsdx = self.model.predict_withGradients(x)
        fmin = self.model.get_fmin()
        phi, Phi, u = get_quantiles(self.jitter, fmin, m, s)    
        f_acqu =  Phi        
        df_acqu = -(phi/s)* (dmdx + dsdx * u)
        return -f_acqu, -df_acqu
    
    
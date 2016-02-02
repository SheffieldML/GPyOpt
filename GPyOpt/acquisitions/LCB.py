from .base import AcquisitionBase
from ..util.general import get_quantiles

class AcquisitionLCB(AcquisitionBase):
    """
    Class for Expected improvement acquisition functions.
    """
    def __init__(self, model, space, exploration_weight= 2, optimizer=None):
        optimizer = optimizer
        super(AcquisitionLCB, self).__init__(model, space, optimizer)
        self.exploration_weight = exploration_weight
    
    def acquisition_function(self,x):
        """
        Expected Improvement
        """
        m, s = self.model.predict(x)
        f_acqu = -m + self.exploration_weight * s
        return -f_acqu  # note: returns negative value for posterior minimization

    def acquisition_function_withGradients(self, x):
        """
        Derivative of the Expected Improvement (has a very easy derivative!)
        """
        m, s, dmdx, dsdx = self.model.predict_withGradients(x)
        f_acqu = -m + self.exploration_weight * s
        df_acqu = -dmdx + self.acquisition_par * dsdx
        return -f_acqu, -df_acqu


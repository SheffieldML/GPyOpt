import numpy as np
from scipy.special import erfc

"""
Acquisition Function classes
"""

from ..util.general import get_moments

class AcquisitionBase(object):
    def __init__(self, acquisition_par=None, invertsign=None):
        self.model = None
        if acquisition_par == None: 
            self.acquisition_par = 0.01
        else: 
            self.acquisition_par = acquisition_par 		
        if invertsign == None: 
            self.sign = 1		
        else: 
            self.sign = -1

    def acquisition_function(self, x):

        pass
    def d_acquisition_function(self, x):
        pass

class AcquisitionEI(AcquisitionBase):
    def acquisition_function(self,x):
        m, s, fmin = get_moments(self.model, x) 	
        u = ((1+self.acquisition_par)*fmin-m)/s	
        phi = np.exp(-0.5 * u**2) / np.sqrt(2*np.pi)
        Phi = 0.5 * erfc(-u / np.sqrt(2))	
        f_acqu = self.sign * (((1+self.acquisition_par)*fmin-m) * Phi + s * phi)
        return -f_acqu  # note: returns negative value for posterior minimization (but we plot +f_acqu)

    def d_acquisition_function(self,x):
        m, s, fmin = get_moments(self.model, x)
        u = ((1+self.acquisition_par)*fmin-m)/s	
        phi = np.exp(-0.5 * u**2) / np.sqrt(2*np.pi)
        Phi = 0.5 * erfc(-u / np.sqrt(2))	
        dmdx, dsdx = self.model.predictive_gradients(x)
        df_acqu =  self.sign* (-dmdx * Phi  + dsdx * phi)
        return -df_acqu
		







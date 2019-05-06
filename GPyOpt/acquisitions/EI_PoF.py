# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from .base import AcquisitionBase
from scipy.special import erf,erfc
from ..util.general import get_quantiles

class AcquisitionEI_PoF(AcquisitionBase):
    """
    Expected improvement acquisition function 
      with Probability of Feasibility  black-box constraint handling

    Based on Gardner et. al. 2014, "Bayesian Optimization with Inequality Constraints"
      also on Gelbart et. al 2014, Gelbart 2015 and Schonlau 1997

    :param model: GPyOpt class of model
    :param model_c: list of GPyOpt class of model 
    :param space: GPyOpt class of domain
    :param optimizer: optimizer of the acquisition. Should be a GPyOpt optimizer
    :param cost_withGradients: function
    :param jitter: positive value to make the acquisition more explorative.
    :param jitter_c: list of positive values to force higher constraint compliance
    :param void_min: positive value to use in case no valid (non-constraint violating) value is available.

    .. Note:: allows to compute the Improvement per unit of cost

    """
    
    analytical_gradient_prediction = True
    
    def __init__(self, model, model_c, space, optimizer=None, cost_withGradients=None, jitter=0.01, jitter_c=None, void_min = 1e5):
        self.model_c = model_c
        super(AcquisitionEI_PoF, self).__init__(model, space, optimizer, cost_withGradients=cost_withGradients)
        self.jitter = jitter
        self.void_min = void_min
        if jitter_c is not None:
            self.jitter_c = jitter_c
        else:
            self.jitter_c = 0.03*np.ones(len(self.model_c))

    @staticmethod
    def fromConfig(model, model_c, space, optimizer, cost_withGradients, config):
        return AcquisitionEI_PoF(model, model_c, space, optimizer, cost_withGradients, 
                                 jitter=config['jitter'],jitter_c=config['jitter_c'],void_min=config['void_min'])
        
    def _compute_acq(self, x):
        """
        Computes the Constrained Expected Improvement per unit of cost
        """
        ########################################################################
        pY, _  = self.model.predict(self.model.model.X)
        pC = np.zeros((pY.shape[0],len(self.model_c)))
        for ic in range(len(self.model_c)):
            pC[:,ic] = self.model_c[ic].model.Y[:,0] 
        
        valid = np.all((pC>0.),axis=1)
        pYv = pY[valid,:]
        if(pYv.shape[0]>0):
            fmin = pYv[:,0].min()
        else:
            fmin = self.void_min
        ########################################################################
                
        m, s = self.model.predict(x)
        
        phi, Phi, u = get_quantiles(self.jitter, fmin, m, s)
        f_acqu = s * (u * Phi + phi)   # constrained f_acqu
        
        ########################################################################
        
        for ic,mdl_c in enumerate(self.model_c):
            m_c, s_c = mdl_c.predict(x)
            
            if isinstance(s_c, np.ndarray):
                s_c[s_c<1e-10] = 1e-10
            elif s_c< 1e-10:
                s_c = 1e-10
            
            z_c = (m_c-self.jitter_c[ic])/s_c    # Implement constraint of type c(x) >= 0
            Phi_c = 0.5*(1+erf(z_c/np.sqrt(2.))) # contrained cdf from erf
            
            f_acqu[...] = f_acqu[...] * Phi_c[...]
        
        ########################################################################

        return f_acqu
    
    def _compute_acq_withGradients(self, x):
        """
        Computes the Constrained Expected Improvement and its derivative (has a not very easy derivative)
        """
        ########################################################################
        pY, _  = self.model.predict(self.model.model.X)
        pC = np.zeros((pY.shape[0],len(self.model_c)))
        for ic in range(len(self.model_c)):
            pC[:,ic] = self.model_c[ic].model.Y[:,0] 
         
        valid = np.all((pC>0.),axis=1)
        pYv = pY[valid,:]
        if(pYv.shape[0]>0):
            fmin = pYv[:,0].min()
        else:
            fmin = self.void_min
        ########################################################################
        
        m, s, dmdx, dsdx = self.model.predict_withGradients(x)
        
        phi, Phi, u = get_quantiles(self.jitter, fmin, m, s)
        f_acqu = s * (u * Phi + phi)
        df_acqu = dsdx * phi - Phi * dmdx
        
        ########################################################################

        Phis_c   = []
        dPFsdx_c = []
        for ic,mdl_c in enumerate(self.model_c):
            m_c, s_c, dmdx_c, dsdx_c = mdl_c.predict_withGradients(x)
            
            if isinstance(s_c, np.ndarray):
                s_c[s_c<1e-10] = 1e-10
            elif s_c< 1e-10:
                s_c = 1e-10
                
            z_c = (m_c-self.jitter_c[ic])/s_c    # Implement constraint of type c(x) >= 0
            phi_c = np.exp(-0.5*(z_c**2))/(np.sqrt(2.*np.pi)*s_c)
            Phi_c = 0.5*(1+erf(z_c/np.sqrt(2.))) # contrained cdf from erf
            dPFsdx = phi_c*(dmdx_c-((m_c-self.jitter_c[ic])/s_c)*dsdx_c)
            
            Phis_c.append(np.copy(Phi_c))
            dPFsdx_c.append(np.copy(dPFsdx))
        
        ########################################################################

        f_acqu_c = np.copy(f_acqu)
        t1 = np.copy(df_acqu)
        t2 = np.zeros(df_acqu.shape)
        for i in range(len(self.model_c)):
            f_acqu_c = f_acqu_c * Phis_c[i]
            t1 = t1 * Phis_c[i]
            
            g2 = np.copy(dPFsdx_c[i])
            for j in range(len(self.model_c)):
                if(j==i):
                    continue
                g2 = g2 * Phis_c[j]
            
            t2 += f_acqu * g2
            
        df_acqu_c = t1+t2
        
        ########################################################################
        
        return f_acqu_c, df_acqu_c
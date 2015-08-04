# Copyright (c) 2015, Javier Gonzalez
# Copyright (c) 2015, the GPy Authors (see GPy AUTHORS.txt)
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
from scipy.stats import norm
from ..util.general import get_moments, get_d_moments, get_quantiles

class AcquisitionBase(object):
    """
    Base class for acquisition functions in Bayesian Optimization
    """
    def __init__(self, acquisition_par=None):
        self.model = None
        if acquisition_par == None: 
            self.acquisition_par = 0.00
        else: 
            self.acquisition_par = acquisition_par
            
    def set_model(self,model):
        self.model = model

    def acquisition_function(self, x):
        pass

    def d_acquisition_function(self, x):
        pass


class AcquisitionEI(AcquisitionBase):
    """
    Class for Expected improvement acquisition functions.
    """
    def acquisition_function(self,x):
        """
        Expected Improvement
        """
        m, s, fmin = get_moments(self.model, x)     
        phi, Phi, _ = get_quantiles(self.acquisition_par, fmin, m, s)    
        f_acqu = (fmin - m + self.acquisition_par) * Phi + s * phi
        return -f_acqu  # note: returns negative value for posterior minimization 

    def d_acquisition_function(self,x):
        """
        Derivative of the Expected Improvement (has a very easy derivative!)
        """
        m, s, fmin = get_moments(self.model, x) 
        dmdx, dsdx = get_d_moments(self.model, x)
        phi, Phi, _ = get_quantiles(self.acquisition_par, fmin, m, s)    
        df_acqu = dsdx * phi - Phi * dmdx
        return -df_acqu
        
class AcquisitionEL(AcquisitionBase):
    """
    Class for acquisition function that accounts for the Expected Loss
    """
    def acquisition_function(self,x):
        """
        1-step ahead expected loss
        """        
        m, s, fmin = get_moments(self.model, x)
        phi, Phi, _ = get_quantiles(self.acquisition_par, fmin, m, s)      # self.acquisition_par should be zero
        loss =  fmin + (m-fmin)*Phi - s*phi                                # same as EI excepting the first term fmin
        return loss
    
    def d_acquisition(self,x):
        """
        Derivative of the 1-step ahead expected loss
        """    
        m, s, fmin = get_moments(self.model, x) 
        dmdx, dsdx = get_d_moments(self.model, x)
        phi, Phi, _ = get_quantiles(self.acquisition_par, fmin, m, s)    
        df_loss = -dsdx * phi + Phi * dmdx                                # same as the EI
        return df_loss

class AcquisitionMPI(AcquisitionBase):
    """
    Class for Maximum Posterior Improvement acquisition functions.
    """
    def acquisition_function(self,x):
        """
        Maximum Posterior Improvement
        """
        m, s, fmin = get_moments(self.model, x)     
        _, Phi,_ = get_quantiles(self.acquisition_par, fmin, m, s)    
        f_acqu =  Phi
        return -f_acqu  # note: returns negative value for posterior minimization 

    def d_acquisition_function(self,x):
        """
        Derivative of the Maximum Posterior Improvement
        """
        m, s, fmin = get_moments(self.model, x) 
        dmdx, dsdx = get_d_moments(self.model, x)
        phi, _, u = get_quantiles(self.acquisition_par, fmin, m, s)    
        df_acqu = -(phi/s)* (dmdx + dsdx + u)
        return -df_acqu


class AcquisitionLCB(AcquisitionBase):
    """
    Class for Upper (lower) Confidence Band acquisition functions.
    """
    def acquisition_function(self,x):
        """
        Upper Confidence Band
        """        
        m, s, _ = get_moments(self.model, x)     
        f_acqu = -m + self.acquisition_par * s
        return -f_acqu  # note: returns negative value for posterior minimization 

    def d_acquisition_function(self,x):
        """
        Derivative of the Upper Confidence Band
        """
        dmdx, dsdx = get_d_moments(self.model, x)
        df_acqu = -dmdx + self.acquisition_par * dsdx
        return -df_acqu

class AcquisitionMP(AcquisitionBase):
    """
    """
    def __init__(self, acq, acquisition_par=None, transform='none'):
        """"""
        super(AcquisitionMP, self).__init__(acquisition_par)
        self.acq = acq
        self.X_batch = None
        self.r_x0=None
        self.s_x0=None
        self.transform=transform.lower()
        if isinstance(acq, AcquisitionLCB) and self.transform=='none':
            self.transform='softplus'
            
    def set_model(self,model):
        self.model = model
        self.acq.model = model

    def update_batches(self, X_batch, L, Min):
        self.X_batch = X_batch
        if X_batch is not None:
            self.r_x0, self.s_x0 = self._hammer_function_precompute(X_batch, L, Min, self.model)
        
    def _hammer_function_precompute(self,x0, L, Min, model):
        if x0 is None: return None, None
        if len(x0.shape)==1: x0 = x0[None,:]
        m = model.predict(x0)[0]
        pred = model.predict(x0)[1].copy()
        pred[pred<1e-16] = 1e-16
        s = np.sqrt(pred)
        r_x0 = (m-Min)/L
        s_x0 = s/L
        r_x0 = r_x0.flatten()
        s_x0 = s_x0.flatten()
        return r_x0, s_x0

    def _hammer_function(self, x,x0,r_x0, s_x0):
        '''
        Creates the function to define the exclusion zones
        '''
        return norm.logcdf((np.sqrt((np.square(np.atleast_2d(x)[:,None,:]-np.atleast_2d(x0)[None,:,:])).sum(-1))- r_x0)/s_x0)

    def _penalized_acquisition(self, x,  model, X_batch, r_x0, s_x0):
        '''
        Creates a penalized acquisition function using 'hammer' functions around the points collected in the batch
        '''
        fval = -self.acq.acquisition_function(x)[:,0]
        
        if self.transform=='softplus':
            fval_org = fval.copy()
            fval[fval_org>=40.] = np.log(fval_org[fval_org>=40.])
            fval[fval_org<40.] = np.log(np.log1p(np.exp(fval_org[fval_org<40.])))
        elif self.transform=='none':
            fval = np.log(fval+1e-50)
        
        fval = -fval
        if X_batch!=None:
            h_vals = self._hammer_function(x, X_batch, r_x0, s_x0)
            fval += -h_vals.sum(axis=-1)
        return fval
    
    def _d_hammer_function(self, x, X_batch, r_x0, s_x0):
        dx = np.atleast_2d(x)[:,None,:]-np.atleast_2d(X_batch)[None,:,:]
        nm = np.sqrt((np.square(dx)).sum(-1))
        z = (nm- r_x0)/s_x0
        h_func = norm.cdf(z)
        
        d = 1./(s_x0*np.sqrt(2*np.pi)*h_func)*np.exp(-np.square(z)/2)/nm
        d[h_func<1e-50] = 0.
        d = d[:,:,None]*dx
        return d.sum(axis=1)

    def acquisition_function(self, x):
        return self._penalized_acquisition(x, self.model, self.X_batch, self.r_x0, self.s_x0)

    def d_acquisition_function(self, x):
        x = np.atleast_2d(x)
        
        if self.transform=='softplus':
            fval = -self.acq.acquisition_function(x)[:,0]
            scale = 1./(np.log1p(np.exp(fval))*(1.+np.exp(-fval)))
        elif self.transform=='none':
            fval = -self.acq.acquisition_function(x)[:,0]
            scale = 1./fval
        else:
            scale = 1.
        
        if self.X_batch is None:
            return scale*self.acq.d_acquisition_function(x)
        else:
            return scale*self.acq.d_acquisition_function(x) - self._d_hammer_function(x, self.X_batch, self.r_x0, self.s_x0)


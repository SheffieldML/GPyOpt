# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from .base import AcquisitionBase
from .LCB import AcquisitionLCB
from .LCB_mcmc import AcquisitionLCB_MCMC
import numpy as np
from scipy.stats import norm

class AcquisitionLP(AcquisitionBase):
    """
    Class for Local Penalization acquisition. Used for batch design.
    :param model: model of the class GPyOpt
    :param space: design space of the class GPyOpt.
    :param optimizer: optimizer of the class GPyOpt.
    :param acquisition: acquisition function of the class GPyOpt
    :param transform: transformation applied to the acquisition (default, none).

    .. Note:: irrespective of the transformation applied the penalized acquisition is always mapped again to the log space.
    This way gradients can be computed additively and are more stable.

    """

    analytical_gradient_prediction = True

    def __init__(self, model, space, optimizer, acquisition, transform='none'):
        super(AcquisitionLP, self).__init__(model, space, optimizer)

        self.acq = acquisition
        self.transform=transform.lower()
        if isinstance(acquisition, AcquisitionLCB) and self.transform=='none':
            self.transform='softplus'
        if isinstance(acquisition, AcquisitionLCB_MCMC) and self.transform=='none':
            self.transform='softplus'

        self.X_batch = None
        self.r_x0=None
        self.s_x0=None

    def update_batches(self, X_batch, L, Min):
        """
        Updates the batches internally and pre-computes the
        """
        self.X_batch = X_batch
        if X_batch is not None:
            self.r_x0, self.s_x0 = self._hammer_function_precompute(X_batch, L, Min, self.model)

    def _hammer_function_precompute(self,x0, L, Min, model):
        """
        Pre-computes the parameters of a penalizer centered at x0.
        """
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

        .. Note:: the penalized acquisition is always mapped to the log space. This way gradients can be computed additively and are more stable.
        '''
        fval = -self.acq.acquisition_function(x)[:,0]

        if self.transform=='softplus':
            fval_org = fval.copy()
            fval[fval_org>=40.] = np.log(fval_org[fval_org>=40.])
            fval[fval_org<40.] = np.log(np.log1p(np.exp(fval_org[fval_org<40.])))
        elif self.transform=='none':
            fval = np.log(fval+1e-50)

        fval = -fval
        if X_batch is not None:
            h_vals = self._hammer_function(x, X_batch, r_x0, s_x0)
            fval += -h_vals.sum(axis=-1)
        return fval

    def _d_hammer_function(self, x, X_batch, r_x0, s_x0):
        """
        Computes the value of the penalizer (centered at x_0) at any x.
        """
        dx = np.atleast_2d(x)[:,None,:]-np.atleast_2d(X_batch)[None,:,:]
        nm = np.sqrt((np.square(dx)).sum(-1))
        z = (nm- r_x0)/s_x0
        h_func = norm.cdf(z)

        d = 1./(s_x0*np.sqrt(2*np.pi)*h_func)*np.exp(-np.square(z)/2)/nm
        d[h_func<1e-50] = 0.
        d = d[:,:,None]
        return d.sum(axis=1)

    def acquisition_function(self, x):
        """
        Returns the value of the acquisition function at x.
        """

        return self._penalized_acquisition(x, self.model, self.X_batch, self.r_x0, self.s_x0)

    def d_acquisition_function(self, x):
        """
        Returns the gradient of the acquisition function at x.
        """
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
            _, grad_acq_x = self.acq.acquisition_function_withGradients(x)
            return scale*grad_acq_x
        else:
            _, grad_acq_x = self.acq.acquisition_function_withGradients(x)
            return scale*grad_acq_x  - self._d_hammer_function(x, self.X_batch, self.r_x0, self.s_x0)

    def acquisition_function_withGradients(self, x):
        """
        Returns the acquisition function and its its gradient at x.
        """
        aqu_x      = self.acquisition_function(x)
        aqu_x_grad = self.d_acquisition_function(x)
        return aqu_x, aqu_x_grad

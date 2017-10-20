# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from .base import EvaluatorBase
import scipy
from ...util.general import samples_multidimensional_uniform
import numpy as np

class LocalPenalization(EvaluatorBase):
    """
    Class for the batch method on 'Batch Bayesian optimization via local penalization' (Gonzalez et al., 2016).

    :param acquisition: acquisition function to be used to compute the batch.
    :param batch size: the number of elements in the batch.

    """
    def __init__(self, acquisition, batch_size):
        super(LocalPenalization, self).__init__(acquisition, batch_size)
        self.acquisition = acquisition
        self.batch_size = batch_size

    def compute_batch(self, duplicate_manager=None, context_manager=None):
        """
        Computes the elements of the batch sequentially by penalizing the acquisition.
        """
        from ...acquisitions import AcquisitionLP
        assert isinstance(self.acquisition, AcquisitionLP)

        self.acquisition.update_batches(None,None,None)

        # --- GET first element in the batch
        X_batch = self.acquisition.optimize()[0]
        k=1

        if self.batch_size >1:
            # ---------- Approximate the constants of the the method
            L = estimate_L(self.acquisition.model.model,self.acquisition.space.get_bounds())
            Min = self.acquisition.model.model.Y.min()

        # --- GET the remaining elements
        while k<self.batch_size:
            self.acquisition.update_batches(X_batch,L,Min)
            new_sample = self.acquisition.optimize()[0]
            X_batch = np.vstack((X_batch,new_sample))
            k +=1

        # --- Back to the non-penalized acquisition
        self.acquisition.update_batches(None,None,None)
        return X_batch


def estimate_L(model,bounds,storehistory=True):
    """
    Estimate the Lipschitz constant of f by taking maximizing the norm of the expectation of the gradient of *f*.
    """
    def df(x,model,x0):
        x = np.atleast_2d(x)
        dmdx,_ = model.predictive_gradients(x)
        res = np.sqrt((dmdx*dmdx).sum(1)) # simply take the norm of the expectation of the gradient
        return -res

    samples = samples_multidimensional_uniform(bounds,500)
    samples = np.vstack([samples,model.X])
    pred_samples = df(samples,model,0)
    x0 = samples[np.argmin(pred_samples)]
    res = scipy.optimize.minimize(df,x0, method='L-BFGS-B',bounds=bounds, args = (model,x0), options = {'maxiter': 200})
    minusL = res.fun[0][0]
    L = -minusL
    if L<1e-7: L=10  ## to avoid problems in cases in which the model is flat.
    return L

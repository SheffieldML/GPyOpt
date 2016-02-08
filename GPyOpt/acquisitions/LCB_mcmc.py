from .LCB import AcquisitionLCB

class AcquisitionLCB_MCMC(AcquisitionLCB):
    """
    Class for Expected improvement acquisition functions.
    """
    def __init__(self, model, space, optimizer=None, cost = None, exploration_weight=2):
        super(AcquisitionLCB_MCMC, self).__init__(model, space, optimizer, cost, exploration_weight)
        
        assert self.model.MCMC_sampler, 'Samples from the hyperparameters are needed to compute the integrated EI'

    def acquisition_function(self,x):
        """
        Expected Improvement
        """    
        means, stds = self.model.predict(x)
        fmin = self.model.get_fmin()
        f_acqu = 0
        for m,s in zip(means, stds):
            f_acqu += self._compute_acq(m, s, x)
        return f_acqu/(len(means))

    def acquisition_function_withGradients(self, x):
        """
        Derivative of the Expected Improvement (has a very easy derivative!)
        """
        means, stds, dmdxs, dsdxs = self.model.predict_withGradients(x)
        fmin = self.model.get_fmin()
        f_acqu = None
        df_acqu = None
        for m, s, dmdx, dsdx in zip(means, stds, dmdxs, dsdxs):
            if f_acqu is None:
                f_acqu, df_acqu = self._compute_acq_withGradients(m, s, dmdx, dsdx, x)
            else:
                f, df = self._compute_acq_withGradients(m, s, dmdx, dsdx, x)
                f_acqu += f
                df_acqu += df
        return f_acqu/(len(means)), df_acqu/(len(means))
from .base import AcquisitionBase
from .LCB import AcquisitionLCB
from ..util.general import get_quantiles, compute_integrated_acquisition, compute_integrated_acquisition_withGradients

class AcquisitionLCB_hmc(AcquisitionBase):
    """
    Class for Expected improvement acquisition functions.
    """
    def __init__(self, model, space, optimizer=None, cost = None, exploration_weight=2):
        optimizer = optimizer
        super(AcquisitionLCB_hmc, self).__init__(model, space, optimizer)
        self.exploration_weight = exploration_weight
        if cost == None: 
            self.cost = lambda x: 1
        else:
            self.cost = cost

        if self.model.num_hmc_samples == None:
            raise Exception('Samples from the hyperparameters are needed to compute the integrated EI')


    def acquisition_function(self,x):
        acquisition = AcquisitionLCB(self.model,self.space,self.optimizer)
        return compute_integrated_acquisition(acquisition,x)


    def acquisition_function_withGradients(self, x):
        acquisition = AcquisitionLCB(self.model,self.space,self.optimizer)
        return compute_integrated_acquisition_withGradients(acquisition,x)
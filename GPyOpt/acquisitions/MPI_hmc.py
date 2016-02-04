from .base import AcquisitionBase
from .MPI import AcquisitionMPI
from ..util.general import get_quantiles, compute_integrated_acquisition, compute_integrated_acquisition_withGradients

class AcquisitionMPI_hmc(AcquisitionBase):
    """
    Class for Expected improvement acquisition functions.
    """
    def __init__(self, model, space, optimizer=None, cost = None, jitter=0.01):
        optimizer = optimizer
        super(AcquisitionMPI_hmc, self).__init__(model, space, optimizer)
        self.jitter = jitter
        if cost == None: 
            self.cost = lambda x: 1
        else:
            self.cost = cost

        if self.model.num_hmc_samples == None:
            raise Exception('Samples from the hyperparameters are needed to compute the integrated EI')


    def acquisition_function(self,x):
        """
        Expected Improvement
        """
        acquisition = AcquisitionMPI(self.model,self.space,self.optimizer)
 
        return compute_integrated_acquisition(acquisition,x)


    def acquisition_function_withGradients(self, x):
        """
        Derivative of the Expected Improvement (has a very easy derivative!)
        """
        acquisition = AcquisitionMPI(self.model,self.space,self.optimizer)
        
        return compute_integrated_acquisition_withGradients(acquisition,x)
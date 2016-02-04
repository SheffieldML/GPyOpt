from .base import AcquisitionBase
from .EI import AcquisitionEI
from ..util.general import get_quantiles

class AcquisitionEI_hmc(AcquisitionBase):
    """
    Class for Expected improvement acquisition functions.
    """
    def __init__(self, model, space, optimizer=None, cost = None, jitter=0.01):
        optimizer = optimizer
        super(AcquisitionEI_hmc, self).__init__(model, space, optimizer)
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
        acqu = AcquisitionEI(self.model,self.space,self.optimizer)
        acqu_x = 0 

        for i in range(self.model.num_hmc_samples): 
            acqu.model.model.kern[:] = self.model.hmc_samples[i,:]
            acqu_x += acqu.acquisition_function(x)

        acqu_x = acqu_x/self.model.num_hmc_samples   
        return -acqu_x/self.cost(x)


    def acquisition_function_withGradients(self, x):
        """
        Derivative of the Expected Improvement (has a very easy derivative!)
        """
        acqu = AcquisitionEI(self.model,self.space,self.optimizer)
        acqu_x = 0 
        d_acqu_x = 0 

        for i in range(self.model.num_hmc_samples):
            acqu.model.model.kern[:] = self.model.hmc_samples[i,:]
            acqu_x += acqu.acquisition_function(x)
            d_acqu_x += acqu.acquisition_function(x)

        acqu_x = acqu_x/self.model.num_hmc_samples  
        d_acqu_x = acqu_x/self.model.num_hmc_samples  
        
        return -acqu_x/self.cost(x), -d_acqu_x/self.cost(x)





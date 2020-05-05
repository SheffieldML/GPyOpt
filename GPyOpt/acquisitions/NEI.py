from GPyOpt.acquisitions.base import AcquisitionBase
from GPyOpt.core.task.cost import constant_cost_withGradients
from ..util.general import get_quantiles
from ..models.gpmodel import GPModel
from torch.quasirandom import SobolEngine
from scipy.stats import norm
import numpy as np
import operator
import functools
from time import time
# from profilehooks import profile

class AcquisitionNEI(AcquisitionBase):
    
    """
    Noisy Expected Improvement with quasi-monte carlo integration acquisition funciton.

    :param model: GPyOpt class of model
    :param space: GPyOpt class of domain
    :param optimizer: optimizer of the acquisition. Should be a GPyOpt optimizer
    :param cost_withGradients: function that provides the evaluation cost and its gradients
    :param QMC_iterations: number of iteration for the Quasi-Monte Carlo integration
    :param jitter: positive value to make the acquisition more explorative

    """

    # --- Set this line to true if analytical gradients are available
    analytical_gradient_prediction = True

    
    def __init__(self, model, space, optimizer, cost_withGradients=None, QMC_iterations = None,jitter = 0, opt_restarts=1):

        self.optimizer = optimizer
        self.QMC_iterations = QMC_iterations
        self.jitter=jitter
        self.batch_elements = []
        self.opt_restarts = opt_restarts
        super(AcquisitionNEI, self).__init__(model, space, optimizer)


    def _compute_acq(self,x,try_parallelize = False):
        # t = time()

        observations = self.model.model.X.tolist()#Concatenate any pending observation to include them in calculation of Expected Improvement.
        for e in self.batch_elements:
            if(len(e)>0):
                observations.append(e.tolist())
        observations = np.array(observations)
        mean,stdv = self.model.predict(observations)
        mean = functools.reduce(operator.iconcat, mean, [])
        stdv = functools.reduce(operator.iconcat, stdv , [])#Flatten m and s (functools.reduce is the most efficient function to do so)
        sobolEngine = SobolEngine(dimension=len(observations),scramble=True)
        sobolSequence = np.array(sobolEngine.draw(self.QMC_iterations))

        NEI = 0
        gpModel = GPModel(None, exact_feval=True,verbose=False, ARD=self.model.ARD,optimize_restarts=self.opt_restarts)
        for k in range(self.QMC_iterations):#quasi monte carlo integration (QMC)
            sobolElement = sobolSequence[k]
            sampled_Y = (norm.ppf(sobolElement,loc=mean,scale=stdv))
            sampled_Y = np.array([np.array([xn]) for xn in sampled_Y])
            gpModel.updateModel(observations,sampled_Y,None,None)
            NEI += self._compute_EI_acq(gpModel,x)/self.QMC_iterations
    
        return NEI
    

    def _compute_acq_withGradients(self, x):

        observations = np.array(self.model.model.X)#Concatenate any pending observation to include them in calculation of Expected Improvement.
        mean,stdv = self.model.predict(observations)
        mean = functools.reduce(operator.iconcat, mean, [])
        stdv = functools.reduce(operator.iconcat, stdv , [])#Flatten m and s (functools.reduce is the most efficient function to do so)
        NEI = 0
        df_NEI=0
        se = SobolEngine(dimension=len(observations),scramble=True)
        sobolSequence = np.array(se.draw(self.QMC_iterations))
        gpModel = GPModel(None, exact_feval=True,verbose=False, ARD=self.model.ARD,optimize_restarts=self.opt_restarts)
        NEI = 0
        df_NEI=0
        for k in range(self.QMC_iterations):#quasi monte carlo integration (QMC)
            sobolElement = sobolSequence[k]
            sampled_Y = (norm.ppf(sobolElement,loc=mean,scale=stdv))
            sampled_Y = np.array([np.array([xn]) for xn in sampled_Y])
            gpModel.updateModel(observations,sampled_Y,None,None)
            partNEI, partdf_NEI = self._compute_EI_acq_withGradients(gpModel,x)
            NEI += partNEI/self.QMC_iterations
            df_NEI += partdf_NEI/self.QMC_iterations

        return NEI, df_NEI


    def _compute_EI_acq(self,model , x):
        """
        Computes the Expected Improvement per unit of cost
        """
        m, s = model.predict(x)
        fmin = model.get_fmin()
        phi, Phi, u = get_quantiles(self.jitter, fmin, m, s)
        f_acqu = s * (u * Phi + phi)
        return f_acqu

    def _compute_EI_acq_withGradients(self, model,x):
        """
        Computes the Expected Improvement and its derivative (has a very easy derivative!)
        """
        fmin = model.get_fmin()
        m, s, dmdx, dsdx = model.predict_withGradients(x)
        phi, Phi, u = get_quantiles(self.jitter, fmin, m, s)
        f_acqu = s * (u * Phi + phi)
        df_acqu = dsdx * phi - Phi * dmdx
        return f_acqu, df_acqu

    def setBatchElements(self,batch_elements):
        #use this function to get multiple experiment before evaluating them.
        #add batch elements that have not been evaluated yet to get a new experiment suggest that takes it into consideration the ones not yet evaluated
        self.batch_elements = batch_elements
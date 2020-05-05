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

    """

    # --- Set this line to true if analytical gradients are available
    analytical_gradient_prediction = True

    
    def __init__(self, model, space, optimizer, cost_withGradients=None, N_QMC = None,jitter = 0, opt_restarts=1):

        self.optimizer = optimizer
        self.N_QMC = N_QMC
        self.jitter=jitter
        self.batch_elements = []
        self.opt_restarts = opt_restarts
        super(AcquisitionNEI, self).__init__(model, space, optimizer)
        
        # --- UNCOMMENT ONE OF THE TWO NEXT BITS
             
        # 1) THIS ONE IF THE EVALUATION COSTS MAKES SENSE
        #
        # if cost_withGradients == None:
        #     self.cost_withGradients = constant_cost_withGradients
        # else:
        #     self.cost_withGradients = cost_withGradients 

        # 2) THIS ONE IF THE EVALUATION COSTS DOES NOT MAKE SENSE
        #
        if cost_withGradients == None:
            self.cost_withGradients = constant_cost_withGradients
        else:
            print('NEI acquisition does now make sense with cost. Cost set to constant.')  
            self.cost_withGradients = constant_cost_withGradients

    # @profile
    def _compute_acq(self,x,try_parallelize = False):
        # t = time()

        observations = self.model.model.X.tolist()#Concatenate any pending observation to include them in calculation of Expected Improvement.
        for e in self.batch_elements:
            if(len(e)>0):
                observations.append(e.tolist())
        observations = np.array(observations)
        m,s = self.model.predict(observations)
        m = functools.reduce(operator.iconcat, m, [])
        s = functools.reduce(operator.iconcat, s , [])#Flatten m and s (functools.reduce is the most efficient function to do so)
        se = SobolEngine(dimension=len(observations),scramble=True)
        tks = np.array(se.draw(self.N_QMC))

        NEI = 0
        gpModel = GPModel(None, exact_feval=True,verbose=False, ARD=self.model.ARD,optimize_restarts=self.opt_restarts)
        for k in range(self.N_QMC):#quasi monte carlo integration (QMC)
            tk = tks[k]
            Fn = (norm.ppf(tk,loc=m,scale=s))
            Fn = np.array([np.array([xn]) for xn in Fn])
            gpModel.updateModel(observations,Fn,None,None)
            NEI += self._compute_EI_acq(gpModel,x)/self.N_QMC
    
        return NEI
    

    def _compute_acq_withGradients(self, x):

        observations = np.array(self.model.model.X)#Concatenate any pending observation to include them in calculation of Expected Improvement.
        m,s = self.model.predict(observations)
        m = functools.reduce(operator.iconcat, m, [])
        s = functools.reduce(operator.iconcat, s , [])#Flatten m and s (functools.reduce is the most efficient function to do so)
        NEI = 0
        df_NEI=0
        se = SobolEngine(dimension=len(observations),scramble=True)
        tks = np.array(se.draw(self.N_QMC))
        gpModel = GPModel(None, exact_feval=True,verbose=False, ARD=self.model.ARD,optimize_restarts=self.opt_restarts)
        NEI = 0
        df_NEI=0
        for k in range(self.N_QMC):#quasi monte carlo integration (QMC)
            tk = tks[k]
            Fn = (norm.ppf(tk,loc=m,scale=s))
            Fn = np.array([np.array([xn]) for xn in Fn])
            gpModel.updateModel(observations,Fn,None,None)
            partNEI, partdf_NEI = self._compute_EI_acq_withGradients(gpModel,x)
            NEI += partNEI/self.N_QMC
            df_NEI += partdf_NEI/self.N_QMC

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
from ...models import GPModel
import numpy as np


class  CostModel(object):
    """
    Class to handle the cost function 
    """
    def __init__(self, cost_withGradients):
        super(CostModel, self).__init__()

        self.cost_type = cost_withGradients
        
        # --- Set-up evaluation cost
        if self.cost_type == None:
            self.cost_withGradients = constant_cost_withGradients
            self.cost_type = 'Constant cost'
                
        elif self.cost_type == 'evaluation_time':
            self.cost_model = GPModel(exact_feval=False,normalize_Y=False,optimize_restarts=5)                                 
            self.cost_withGradients  = self._cost_gp_withGradients   
            self.num_updates = 0
        else: 
            self.cost_withGradients  = cost_withGradients
            self.cost_type  = 'Used defined cost'


    def _cost_gp(self,x):
        m, _, _, _= self.cost_model.predict_withGradients(x)
        return np.exp(m)

    def _cost_gp_withGradients(self,x):
            m, _, dmdx, _= self.cost_model.predict_withGradients(x)
            return np.exp(m), np.exp(m)*dmdx

    def update_cost_model(self, x, cost_x):

        if self.cost_type == 'evaluation_time':
            cost_evals = np.log(np.atleast_2d(np.asarray(cost_x)).T)

            if self.num_updates == 0:
                X_all = x
                costs_all = cost_evals
            else:
                X_all = np.vstack((self.cost_model.model.X,x))
                costs_all = np.vstack((self.cost_model.model.Y,cost_evals))
            
            self.num_updates += 1
            self.cost_model.updateModel(X_all, costs_all, None, None)

def constant_cost_withGradients(x):
        return np.ones(x.shape[0])[:,None], np.zeros(x.shape)




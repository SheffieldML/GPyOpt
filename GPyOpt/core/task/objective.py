import time
import numpy as np
from ...util.general import spawn
from ...util.general import get_d_moments
import GPy
import GPyOpt

class Objective(object):
    
    def evaluate(self, x):
        pass

class SingleObjective(Objective):
    
    def __init__(self, func, space, cost = None, cost_grad=None):
        self.func  = func
        self.space = space
        self.cost_type = cost
        
        # No cost used
        if self.cost_type == None:
            cost = lambda x: np.ones(x.shape[0])[:,None]   
            self.cost_grad = lambda x: np.zeros(x.shape)
        
        # Function evaluation time used as cost
        elif self.cost_type == 'computing_time':
             self.cost_model = GPyOpt.models.GPModel(exact_feval=False,normalize_Y=False,optimize_restarts=5)
             self.cost = self._cost_gp
             self.cost_grad = self._cost_gp_grad

        # Explicit cost defined by the user
        else: 
            self.cost      = cost
            self.cost_grad = cost_grad

        self.n_evals = 0
        
    def _cost_gp(self,x):
        return self.cost_model.model.predict(x)[0]

    def _cost_gp_grad(self,x):
        dmdx, _ = self.cost_model.model.predictive_gradients(x)
        return dmdx[:,:,0] 

    def evaluate(self, x):        
        cost_evals = []
        f_evals     = np.empty(shape=[0, 1])
        
        for i in range(x.shape[0]): 
            st_time    = time.time()
            f_evals     = np.vstack([f_evals,self.func(np.atleast_2d(x[i]))])
            cost_evals += [time.time()-st_time]     
        
        if self.cost_type == 'computing_time':
            cost_evals = np.log(np.atleast_2d(np.asarray(cost_evals)).T)

            if self.n_evals == 0:
                X_all = x
                costs_all = cost_evals
                self.n_evals = 1
            else:
                X_all = np.vstack((self.cost_model.model.X,x))
                costs_all = np.vstack((self.cost_model.model.Y,cost_evals))
            self.cost_model.updateModel(X_all, costs_all, None, None)
        return f_evals, cost_evals 
    
    def evalute_cost(self,x):
        return np.exp(self.cost_model.model.predict(x)[0])


class SingleObjectiveMultiProcess(SingleObjective):

    def __init__(self, func, space, n_procs=2, batch_eval=True):
        super(SingleObjectiveMultiProcess, self).__init__(func, space, batch_eval)
        self.n_procs = n_procs
        
    def evaluate(self, x):
        st_time = time.time()
        try:
            # --- Parallel evaluation of *f* if several cores are available
            from multiprocessing import Process, Pipe
            from itertools import izip          
            divided_samples = [x[i::self.n_procs] for i in xrange(self.n_procs)]
            pipe=[Pipe() for i in xrange(self.n_procs)]
            proc=[Process(target=spawn(self.func),args=(c,x)) for x,(p,c) in izip(divided_samples,pipe)]
            [p.start() for p in proc]
            [p.join() for p in proc]
            res = np.vstack([p.recv() for (p,c) in pipe])
        except:
            if not hasattr(self, 'parallel_error'):
                print 'Error in parallel computation. Fall back to single process!'
                self.parallel_error = True 
            res = self.func(x)
        return res, time.time()-st_time

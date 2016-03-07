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
    
    def __init__(self, func, batch_size = 1, num_cores = 1, objective_name = 'no_name', batch_type = 'syncronous', space = None):
        self.func  = func
        self.batch_size = batch_size
        self.n_procs = num_cores
        self.num_evaluations = 0
        self.space = space
        self.objective_name = objective_name


    def evaluate(self, x):        

        if self.batch_size == 1:
            f_evals, cost_evals = self._single_evaluation(x)
        else:
            try:
                f_evals, cost_evals = self._syncronous_batch_evaluation(x)
            except:
                if not hasattr(self, 'parallel_error'):
                    print 'Error in parallel computation. Fall back to single process!'
                    f_evals, cost_evals = self._single_evaluation(x)        
        return f_evals, cost_evals 


    def _single_evaluation(self,x):
        cost_evals = []
        f_evals     = np.empty(shape=[0, 1])
        
        for i in range(x.shape[0]): 
            st_time    = time.time()
            f_evals     = np.vstack([f_evals,self.func(np.atleast_2d(x[i]))])
            cost_evals += [time.time()-st_time]  
        return f_evals, cost_evals 


    def _syncronous_batch_evaluation(self,x):   
        from multiprocessing import Process, Pipe
        from itertools import izip          
        
        # --- parallel evaluation of the function
        divided_samples = [x[i::self.n_procs] for i in xrange(self.n_procs)]
        pipe = [Pipe() for i in xrange(self.n_procs)]
        proc = [Process(target=spawn(self.func),args=(c,x)) for x,(p,c) in izip(divided_samples,pipe)]
        [p.start() for p in proc]
        [p.join() for p in proc] 
        f_evals = np.vstack([p.recv() for (p,c) in pipe])
        
        # --- time of evalation is set to constant (=1). This is one of the hypothesis of syncronous batch methods.
        cost_evals = np.ones((x.shape[0],1))
        return f_evals, cost_evals 

    def _asyncronous_batch_evaluation(self,x):   
        ### --- TODO
        pass


 






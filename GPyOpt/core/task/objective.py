
import time
import numpy as np
from ...util.general import spawn
import GPy

class Objective(object):
    
    def evaluate(self, x):
        pass

# class SingleObjective(Objective):
    
#     def __init__(self, func, space, batch_eval=True):
#         self.func  = func
#         self.space = space
#         self.batch_eval = batch_eval
        
#     def evaluate(self, x):
#         st_time = time.time()
#         if self.batch_eval:
#             res = self.func(x)
#         else:
#             res = np.hstack([self.func(x[i]) for i in range(x.shape[0])])[:,None]
#         return res, time.time()-st_time
    
# class SingleObjective(Objective):
    
#     def __init__(self, func, space, cost = 'False'):
#         self.func  = func
#         self.space = space
        
        
#     def evaluate(self, x):
#         st_time = time.time()
#         time_cost = []
#         res = np.empty(shape=[0, 1])
#         for i in range(x.shape[0]): 
#             st_time = time.time()
#             res = np.vstack([res,self.func(np.atleast_2d(x[i]))])
#             time_cost += [time.time()-st_time]
#         return res, np.asarray(time_cost)


class SingleObjective(Objective):
    
    def __init__(self, func, space, cost = None):
        self.func  = func
        self.space = space
        self.cost_type = cost
        
        if self.cost_type == None:
            self.cost = lambda x: 1
            
        #elif self.cost_type = 'time':
            # TODO create GP model
        #    self.GPcost = GPy.models.GPRegression 
        #    self.cost = None
        else: 
            self.cost = cost
            
        
    def evaluate(self, x):
        st_time = time.time()
        
        cost_evals = []
        f_evals     = np.empty(shape=[0, 1])
        
        for i in range(x.shape[0]): 
            st_time    = time.time()
            f_evals     = np.vstack([f_evals,self.func(np.atleast_2d(x[i]))])
            cost_evals += [time.time()-st_time]
        
        cost_evals = np.asarray(cost_evals)
        
        #if self.cost_type == 'time'
            # TODO: update GP model on the log of the times
            # self.GPcost....    
            
        return f_evals, cost_evals
    
    #def evalute_cost(self,x):
    #    return self.GPcost.predict(x)[0]


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

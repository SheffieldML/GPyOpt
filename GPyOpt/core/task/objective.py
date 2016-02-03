
import time
import numpy as np
from ...util.general import spawn

class Objective(object):
    
    def evaluate(self, x):
        pass
    

class SingleObjective(Objective):
    
    def __init__(self, func, space, batch_eval=True):
        self.func  = func
        self.space = space
        self.batch_eval = batch_eval
        
    def evaluate(self, x):
        st_time = time.time()
        if self.batch_eval:
            res = self.func(x)
        else:
            res = np.hstack([self.func(x[i]) for i in range(x.shape[0])])[:,None]
        return res, time.time()-st_time
    
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

# class Objective(object):
#     """
#     This is a class to handle the objective function, its associated cost and the way it should be evaluated.

#     - func: is a list of functions to be optimized. It has only one element in the most general case.
#     - cost: is a lisy of costs of evaluating the functions to optimize. In principle it will be a deterministic function but 
#     we can extend this and handle it with the log of a GP.
#     - evaluation_type: determine how the function is going to be evaluated ['sequentiely','syncr_batch','async_batch']

#     """
#     def __init__(self, func, cost = None, evaluation_type = None):
#         if len(func)==1:
#             self.type = 'standard' 
#             self.f = {'f1': func}
            
#         else: 
#             self.type = 'multi-task'
#             self.f = {}
#             for i in len(func):
#                 self.f['f'+str(i+1)] = func[i]

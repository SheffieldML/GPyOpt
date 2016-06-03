# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import time
import numpy as np
from ...util.general import spawn
from ...util.general import get_d_moments
import GPy
import GPyOpt

class Objective(object):
    """
    General class to handle the objective function internaly.
    """
    
    def evaluate(self, x):
        pass


class SingleObjective(Objective):
    """
    Class to hadle problems with one single objective function.

    param func: objective fuction. 
    param batch_size: size of the batches (default, 1)
    param num_cores: number of cores to use in the process of evaluting the objective (default, 1).
    param objective_name: name of the objective function.
    param batch_type: Type of batch used. Only 'syncronous' evaluations are posible at the moment.
    param space: Not in use.

    .. Note:: the objective function should take 2-dimentionnal numpy arrays as input and outputs. Each row should
    contain a location (in the case of the inputs) or a function evalution (in the case of the outputs).
    """

    
    def __init__(self, func, batch_size = 1, num_cores = 1, objective_name = 'no_name', batch_type = 'syncronous', space = None):
        self.func  = func
        self.batch_size = batch_size
        self.n_procs = num_cores
        self.num_evaluations = 0
        self.space = space
        self.objective_name = objective_name


    def evaluate(self, x):       
        """
        Performs the evalution of the objective at x.
        """

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
        """
        Performs sequential evaluations of the function at x (single location or batch). The computing time of each 
        evaluation is also provided.
        """
        cost_evals = []
        f_evals     = np.empty(shape=[0, 1])
        
        for i in range(x.shape[0]): 
            st_time    = time.time()
            f_evals     = np.vstack([f_evals,self.func(np.atleast_2d(x[i]))])
            cost_evals += [time.time()-st_time]  
        return f_evals, cost_evals 


    def _syncronous_batch_evaluation(self,x):   
        """
        Evalutes the function a x, where x can be a single location or a batch. The evalution is performed in parallel
        according to the number of accesible cores.
        """
        from multiprocessing import Process, Pipe
        from itertools import izip          
        
        # --- parallel evaluation of the function
        divided_samples = [x[i::self.n_procs] for i in range(self.n_procs)]
        pipe = [Pipe() for i in range(self.n_procs)]
        proc = [Process(target=spawn(self.func),args=(c,k)) for k,(p,c) in izip(divided_samples,pipe)]
        [p.start() for p in proc]
        [p.join() for p in proc] 
        
        # --- time of evalation is set to constant (=1). This is one of the hypothesis of syncronous batch methods.
        f_evals = np.zeros((x.shape[0],1))
        cost_evals = np.ones((x.shape[0],1))
        i = 0
        for (p,c) in pipe:
            f_evals[i::self.n_procs] = p.recv()
            i += 1
        return f_evals, cost_evals 

    def _asyncronous_batch_evaluation(self,x):   

        """
        Performs the evalution of the function at x while other evaluations are pending. 
        """
        ### --- TODO
        pass


 






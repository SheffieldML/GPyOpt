# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import time
import numpy as np

class Objective(object):
    """
    General class to handle the objective function internally.
    """

    def evaluate(self, x):
        raise NotImplementedError()


class SingleObjective(Objective):
    """
    Class to handle problems with one single objective function.

    param func: objective function.
    param batch_size: size of the batches (default, 1)
    param num_cores: number of cores to use in the process of evaluating the objective (default, 1).
    param objective_name: name of the objective function.
    param batch_type: Type of batch used. Only 'synchronous' evaluations are possible at the moment.
    param space: Not in use.

    .. Note:: the objective function should take 2-dimensional numpy arrays as input and outputs. Each row should
    contain a location (in the case of the inputs) or a function evaluation (in the case of the outputs).
    """


    def __init__(self, func, num_cores = 1, objective_name = 'no_name', batch_type = 'synchronous', space = None):
        self.func  = func
        self.n_procs = num_cores
        self.num_evaluations = 0
        self.space = space
        self.objective_name = objective_name


    def evaluate(self, x):
        """
        Performs the evaluation of the objective at x.
        """

        if self.n_procs == 1 or x.shape[0] == 1:
            f_evals, cost_evals = self._eval_func(x)
        else:
            f_evals, cost_evals = self._syncronous_batch_evaluation(x)

        return f_evals, cost_evals


    def _eval_func(self, x):
        """
        Performs sequential evaluations of the function at x (single location or batch). The computing time of each
        evaluation is also provided.
        """
        cost_evals = []
        f_evals     = np.empty(shape=[0, 1])

        for i in range(x.shape[0]):
            st_time    = time.time()
            rlt = self.func(np.atleast_2d(x[i]))
            f_evals     = np.vstack([f_evals,rlt])
            cost_evals += [time.time()-st_time]
        return f_evals, cost_evals


    def _syncronous_batch_evaluation(self,x):
        """
        Evaluates the function a x, where x can be a single location or a batch. The evaluation is performed in parallel
        according to the number of accessible cores.
        """
        from sys import version_info
        # --- parallel evaluation of the function
        divided_samples = [x[i::self.n_procs] for i in range(self.n_procs)]

        if version_info.major < 3:
            from multiprocess import Pool
            p = Pool(processes=self.n_procs)
            results = p.map(self._eval_func, divided_samples)
            p.close()
            p.join()
        else:
            from joblib import Parallel, delayed
            results = Parallel(n_jobs=self.n_procs)([delayed(self._eval_func)(args) for args in divided_samples])

        f_evals = np.vstack([res[0] for res in results])
        cost_evals = np.hstack([res[1] for res in results])

        return f_evals, cost_evals

    def _asyncronous_batch_evaluation(self,x):

        """
        Performs the evaluation of the function at x while other evaluations are pending.
        """
        ### --- TODO
        pass

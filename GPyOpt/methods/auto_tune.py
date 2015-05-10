# Copyright (c) 2015, Javier Gonzalez
# Copyright (c) 2015, the GPy Authors (see GPy AUTHORS.txt)
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from .bayesian_optimization import BayesianOptimization

class autoTune(BayesianOptimization):
    def __init__(self, f, bounds=None, max_iter=None, eps = None, n_procs = 1, report_file = None):
        '''
        Automatic parameter tuner for computer models based on the GPyOPt class BayesianOptimization. Note that:
            (1) The function is mainly tuned to provide accurate results so it may be slow is expensive examples.
            (2) It is recomended to used as many processors are available for the opimization. This may significantly improve the resutls.
            (3) This function depends on DIRECT, which is use to optimize the aquisition function.
        :param *f* the function to optimize. Should get a nxp numpy array as input and return a nx1 numpy array.
        :param bounds: Tuple containing the box constrains of the function to optimize. Example: for [0,1]x[0,1] insert [(0,1),(0,1)].
        :param max_iter: exploration horizon, or number of acquisitions. It nothing is provided optimizes the current acquisition.  
        :param eps: minimum distance between two consecutive x's to keep running the model.
        :param n_procs: number of CPUs to use in computation. Is set but default to be equal to the size of the betches collected.
        :param report_file: name of the file in which the results of the optimization are saved.
        '''
        input_dim       = len(bounds)

        # ---- Initial number of data points
        self.model_data_init = 5*input_dim

        # ---- Maximum number of iterations
        if max_iter==None: 
            max_iter = 15*input_dim

        # ---- Parallel computation
        n_inbatch = n_procs

        # ---- tolerance
        if eps==None: 
            self.eps = 1e-6
        else:
            self.eps = eps

        # ---- File to save the report 
        if report_file==None: 
            self.report_file='GPyOpt-results.txt'
        else:
            self.report_file=report_file	

        # ----- Asign super class 
        super(autoTune,self).__init__(f,bounds,exact_feval=True)

        # ----  Run optimization. Customized for automatic tuning. Reinforces the acquisition with a random location.
        self.run_optimization(  max_iter = max_iter, 
                                n_inbatch=n_inbatch, 
                                acqu_optimize_method='DIRECT', 
                                acqu_optimize_restarts=200, 
                                batch_method='random', 
                                eps = eps, 
                                n_procs=n_procs, 
                                true_gradients = True, 
                                save_interval=5, 
                                report_file = report_file, 
                                verbose=True) 




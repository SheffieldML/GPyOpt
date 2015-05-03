# Copyright (c) 2015, Javier Gonzalez
# Copyright (c) 2015, the GPy Authors (see GPy AUTHORS.txt)
# Licensed under the BSD 3-clause license (see LICENSE.txt)


from .bayesian_optimization import BayesianOptimization

class autoTune(BayesianOptimization):
    def __init__(self, f, bounds=None, max_iter=None, eps = None, n_procs = 1, report_file = None, plot_file = None):
        '''
        Automatic parameter tuner for computer models based on the GPyOPt classs BayesianOptimization
        '''
        input_dim       = len(bounds)

        # ---- Initial number of data points
        model_data_init = 10*input_dim

        # ---- Maximum number of iterations
        if max_iter==None: 
            max_iter = 10*input_dim

        # ---- Parallel computation
        n_batch = n_procs

        # ---- tolerance
        if eps==None: 
            self.eps = 1e-6
        else:
            self.eps = eps

        #  ---- File to save the report 
        if report_file==None: 
            self.report_file='GPyOpt_results.txt'
        
        # ----  File to save the diagnostics plot
        if plot_file==None: 
            plot_file='GPyOpt_diagnostic_plot.pdf'

        # ----- Asign super class 
        super(autoTune,self).__init__(f,bounds,exact_feval=True)

        # ----  Run optimization. Customized for automatic tuning. Reinforces the acquisition with a random location.
        self.run_optimization(  max_iter = max_iter, 
                                n_inbatch=2, 
                                acqu_optimize_method='DIRECT', 
                                acqu_optimize_restarts=200, 
                                batch_method='random', 
                                eps = eps, 
                                n_procs=2, 
                                true_gradients = True, 
                                save_interval=5, 
                                report_file = report_file, 
                                verbose=True) 
        
        # TODO
        # self.plot_convergence(plot_file)
	



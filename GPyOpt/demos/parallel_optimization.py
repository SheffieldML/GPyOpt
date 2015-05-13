# Copyright (c) 2015, Javier Gonzalez
# Copyright (c) 2015, the GPy Authors (see GPy AUTHORS.txt)
# Licensed under the BSD 3-clause license (see LICENSE.txt)


"""
This is a demo to demonstrate how to perform parallel Bayesian optimization with GPyOpt. Run the example by writing:

import GPyOpt
BO_demo_parallel= GPyOpt.demos.parallel_optimization()

As a result you should see:

- A plot with the model and the current acquisition function
- A plot with the diagnostic plots of the optimization.
- An object call BO_demo_auto that contains the results of the optimization process (see reference manual for details). Among the available results you have access to the GP model via

>> BO_demo_parallel.model

and to the location of the best found location writing.

BO_demo_parallel.x_opt

"""

def parallel_optimization(plots=True):
    import GPyOpt
    from numpy.random import seed
    seed(12345)
    
    # --- Objective function
    objective_true  = GPyOpt.fmodels.experiments2d.branin()                 # true function
    objective_noisy = GPyOpt.fmodels.experiments2d.branin(sd = 0.1)         # noisy version
    bounds = objective_noisy.bounds                                         # problem constrains 

    # --- Problem definition and optimization
    BO_demo_parallel = GPyOpt.methods.BayesianOptimization(f=objective_noisy.f,  # function to optimize       
                                            bounds = bounds,                     # box-constrains of the problem
                                            acquisition = 'EI',                 # Selects the Expected improvement
                                            acquisition_par = 0,                 # parameter of the acquisition function
                                            normalize = True)                    # Normalize the acquisition function
    
    
    # --- Run the optimization
    max_iter = 15                                                          

    # --- Number of cores to use in the optimization (parallel evaluations of f)
    n_cores = 3

    print '-----'
    print '----- Running demo. It may take a few seconds.'
    print '-----'
    
    # --- Run the optimization                                              # evaluation budget
    BO_demo_parallel.run_optimization(max_iter,                             # Number of iterations
                                acqu_optimize_method = 'fast_random',       # method to optimize the acq. function
                                n_inbatch = n_cores,                        # size of the collected batches (= number of cores)
                                batch_method='mp',                          # method to collected the batches (maximization-penalization)
                                acqu_optimize_restarts = 30,                # number of local optimizers
                                eps = 10e-6)                                # secondary stop criteria (apart from the number of iterations) 

    # --- Plots
    if plots:
        objective_true.plot()
        BO_demo_parallel.plot_acquisition()
        BO_demo_parallel.plot_convergence()
        
    
    return BO_demo_parallel 

# Copyright (c) 2015, Javier Gonzalez
# Copyright (c) 2015, the GPy Authors (see GPy AUTHORS.txt)
# Licensed under the BSD 3-clause license (see LICENSE.txt)


"""
This is a simple demo to demonstrate the use of Bayesian optimization with GPyOpt with some simple options. Run the example by writing:

import GPyOpt
BO_demo_1d = GPyOpt.demos.begginer_optimization_1d()

As a result you should see:

- A plot with the model and the current acquisition function
- A plot with the diagnostic plots of the optimization.
- An object call BO_demo_auto that contains the results of the optimization process (see reference manual for details). Among the available results you have access to the GP model via

>> BO_demo_1d.model

and to the location of the best found location writing.

BO_demo_auto.x_opt

"""

def begginer_optimization_1d(plots=True):
    import GPyOpt
    from numpy.random import seed
    seed(12345)
    
    # --- Objective function
    objective_true  = GPyOpt.fmodels.experiments1d.forrester()              # true function
    objective_noisy = GPyOpt.fmodels.experiments1d.forrester(sd= .25)       # noisy version
    bounds = [(0,1)]                                                        # problem constrains 

    # --- Problem definition and optimization
    BO_demo_1d = GPyOpt.methods.BayesianOptimization(f=objective_noisy.f,   # function to optimize       
                                             bounds=bounds,                 # box-constrains of the problem
                                             acquisition='EI')              # Selects the Expected improvement
    # Run the optimization
    max_iter = 15                                                           

    print '-----'
    print '----- Running demo. It may take a few seconds.'
    print '-----'
    
    # Run the optimization                                                  
    BO_demo_1d.run_optimization(max_iter,                                  # evaluation budget
                                    eps=10e-6)                   # stop criterion
                            

    # --- Plots
    if plots:
        objective_true.plot()
        BO_demo_1d.plot_acquisition()
        BO_demo_1d.plot_convergence()
        
    
    return BO_demo_1d 
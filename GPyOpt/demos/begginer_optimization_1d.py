# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

"""
This is a simple demo to demonstrate the use of Bayesian optimization with GPyOpt with some simple options. Run the example by writing:

import GPyOpt
BO_demo_1d = GPyOpt.demos.begginer_optimization_1d()

As a result you should see:

- A plot with the model and the current acquisition function
- A plot with the diagnostic plots of the optimization.
- An object call BO_demo_1d that contains the results of the optimization process (see reference manual for details). Among the available results you have access to the GP model via

>> BO_demo_1d.model

and to the location of the best found location writing.

BO_demo_1d.x_opt

"""

def begginer_optimization_1d(plots=True):
    import GPyOpt
    from numpy.random import seed
    seed(1234)
        

    # Create the true and perturbed Forrester function and the boundaries of the problem
    f_true= GPyOpt.objective_examples.experiments1d.forrester()          
    bounds = [{'name': 'var_1', 'type': 'continuous', 'domain': (0,1)}]  

    myBopt = GPyOpt.methods.BayesianOptimization(f=f_true.f,                
                                                domain=bounds,        
                                                acquisition_type='EI',
                                                exact_feval = True)

    # --- Problem definition and optimization
    max_time = 10        
    eps = 1e-8                                                 

    print '-----'
    print '----- Running demo. It may take a few seconds.'
    print '-----'
    
    # Run the optimization                                                  
    myBopt.run_optimization(max_time=max_time, eps=eps)   
                            

    # --- Plots
    if plots:
        f_true.plot()
        myBopt.plot_acquisition()
        myBopt.plot_convergence()
        
    return myBopt
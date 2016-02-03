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
- An object call BO_demo_1d that contains the results of the optimization process (see reference manual for details). Among the available results you have access to the GP model via

>> BO_demo_1d.model

and to the location of the best found location writing.

BO_demo_1d.x_opt

"""

def begginer_optimization_1d(plots=True):
    import GPyOpt
    from numpy.random import seed
    seed(1234)
    
    # --- Objective function
    objective_true  = GPyOpt.fmodels.experiments1d.forrester()              # true function
    objective_noisy = GPyOpt.fmodels.experiments1d.forrester(sd= .25)       # noisy version
    bounds = [(0,1)]                                                        # problem constrains 


    # --- Problem definition and optimization
    BO_demo_1d = GPyOpt.methods.BayesianOptimization(f=objective_noisy.f,   # function to optimize       
                                                    bounds=bounds,          # box-constrains of the problem
                                                    acquisition_type='EI')       # Selects the Expected improvement
    # Run the optimization for 10 seconds
    max_time = 10                                                         

    print '-----'
    print '----- Running demo. It may take a few seconds.'
    print '-----'
    
    # Run the optimization                                                  
    BO_demo_1d.run_optimization(max_time=max_time,                                   # evaluation budget
                                    eps=10e-6)                              # stop criterion
                            

    # --- Plots
    if plots:
        objective_true.plot()
        BO_demo_1d.plot_acquisition()
        BO_demo_1d.plot_convergence()
        
    return BO_demo_1d 

def simple_optimization_1d(plots=True):
    import GPyOpt
    from numpy.random import seed
    seed(1234)
        
    # --- Objective function
    objective_true  = GPyOpt.objective_examples.experiments1d.forrester()              # true function
    objective_noisy = GPyOpt.objective_examples.experiments1d.forrester(sd= .25)       # noisy version
    bounds = [(0,1)]                                                        # problem constrains 

    model = GPyOpt.models.GPModel(exact_feval=True, optimize_restarts=10)
    space = GPyOpt.Design_space([{'domain':(0,1)}])
    obj = GPyOpt.core.task.SingleObjective(objective_true.f, space)

    opt = GPyOpt.optimization.ContAcqOptimizer(space, 50)
    acq = GPyOpt.acquisitions.AcquisitionEI(model, space, optimizer=opt)
    
    X_init = GPyOpt.util.stats.initial_design('random', space.get_continuous_bounds(), 2)
    
    bo = GPyOpt.core.BO(model, space, obj, acq, X_init)

    # --- Problem definition and optimization
    max_time = 10                                                         

    print '-----'
    print '----- Running demo. It may take a few seconds.'
    print '-----'
    
    # Run the optimization                                                  
    bo.run_optimization(max_time=max_time,                                   # evaluation budget
                                    eps=1e-8)                              # stop criterion
                            

    # --- Plots
    if plots:
        objective_true.plot()
        bo.plot_acquisition()
        bo.plot_convergence()
        
    return bo 
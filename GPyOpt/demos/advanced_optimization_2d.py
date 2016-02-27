# Copyright (c) 2015, Javier Gonzalez
# Copyright (c) 2015, the GPy Authors (see GPy AUTHORS.txt)
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from __future__ import print_function

"""
This is a simple demo to demonstrate the use of Bayesian optimization with GPyOpt with some simple options. Run the example by writing:

import GPyOpt
BO_demo_2d = GPyOpt.demos.advanced_optimization_2d()

As a result you should see:

- A plot with the model and the current acquisition function
- A plot with the diagnostic plots of the optimization.
- An object call BO_demo_auto that contains the results of the optimization process (see reference manual for details). Among the available results you have access to the GP model via

>> BO_demo_2d.model

and to the location of the best found location writing.

BO_demo_2d.x_opt

"""

def advanced_optimization_2d(plots=True):
    import GPyOpt
    import GPy
    from numpy.random import seed
    seed(12345)



    # --- Objective function
    objective_true  = GPyOpt.fmodels.experiments2d.sixhumpcamel()             # true function
    objective_noisy = GPyOpt.fmodels.experiments2d.sixhumpcamel(sd = 0.1)     # noisy version
    bounds = objective_noisy.bounds                                           # problem constrains
    input_dim = len(bounds)

    # Select an specific kernel from GPy
    kernel = GPy.kern.RBF(input_dim, variance=.1, lengthscale=.1) + GPy.kern.Bias(input_dim) # we add a bias kernel


    # --- Problem definition and optimization
    BO_demo_2d = GPyOpt.methods.BayesianOptimization(f=objective_noisy.f,  # function to optimize
                                            kernel = kernel,               # pre-specified model
                                            bounds=bounds,                 # box-constrains of the problem
                                            acquisition='LCB',             # Selects the Expected improvement
                                            acquisition_par = 2,           # parameter of the acquisition function
                                            numdata_initial_design = 15,    # 15 initial points
                                            type_initial_design='latin',   # latin desing of the initial points
                                            model_optimize_interval= 2,    # The model is updated every two points are collected
                                            normalize = True)              # normalized y


    # Run the optimization
    max_iter = 20

    print('-----')
    print('----- Running demo. It may take a few seconds.')
    print('-----')

    # --- Run the optimization                                              # evaluation budget
    BO_demo_2d.run_optimization(max_iter,                                   # Number of iterations
                                acqu_optimize_method = 'DIRECT',       # method to optimize the acq. function
                                acqu_optimize_restarts = 30,                # number of local optimizers
                                eps=10e-6,                        # secondary stop criteria (apart from the number of iterations)
                                true_gradients = True)                     # The gradients of the acquisition function are approximated (faster)


    # --- Plots
    if plots:
        objective_true.plot()
        BO_demo_2d.plot_acquisition()
        BO_demo_2d.plot_convergence()


    return BO_demo_2d

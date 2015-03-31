# Copyright (c) 2015, Javier Gonzalez
# Copyright (c) 2015, the GPy Authors (see GPy AUTHORS.txt)
# Licensed under the BSD 3-clause license (see LICENSE.txt)


"""
This is a simple demo to demonstrate the use of Bayesian optimization for automatic
optimization of black-box functions. All the default values of the parameters in GPyOPt 
are used. Run the example by writing:

>> import GPyOpt
>> BO_demo_auto = GPyOpt.demos.automatic_optimization()

As a result you should see:

- A plot with the model and the current acquisition function, the Expected Improvement in this example.#
- A plot with the diagnostic plots of the optimization.
- An object call BO_demo_auto that contains the results of the optimization process (see reference manual for details). 
Among the available results you have access to the GP model via

>> BO_demo_auto.model

and to the location of the best found location writing.

BO_demo_auto.x_opt

"""

def automatic_optimization(plots=True):
    import GPyOpt
    from numpy.random import seed
    seed(12345)
    
    # --- Objective function
    def objective(x):
        return (2*x)**2

    # --- Bounds
    bounds = [(-1,1)]

    print '-----'
    print '----- Running demo. It may take a few seconds.'
    print '-----'

    # --- Problem definition and optimization
    BO_demo_auto = GPyOpt.methods.BayesianOptimization(objective,bounds)
    BO_demo_auto.run_optimization()

    # --- Plots
    if plots:
        BO_demo_auto.plot_acquisition()
        BO_demo_auto.plot_convergence()
    
    return BO_demo_auto 
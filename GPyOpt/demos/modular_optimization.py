def modular_optimization(plots=True):
    import GPyOpt
    from numpy.random import seed
    seed(1234)
        
    # --- Fucntion to optimize
    func  = GPyOpt.objective_examples.experiments1d.forrester(sd=0.05)  
    #cost = lambda x: 2*x

    # --- Space design
    space = GPyOpt.Design_space([{'domain':(0,1)}])

    # --- Objective
    objective = GPyOpt.core.task.SingleObjective(func.f, space)

    # --- CHOOSE the model type
    model = GPyOpt.models.GPModel(exact_feval=True, num_hmc_samples=50)
    
    # --- CHOOSE the acquisition optimizer
    aquisition_optimizer = GPyOpt.optimization.ContAcqOptimizer(space, 500)
    
    # --- CHOOSE the type of acquisition
    acquisition = GPyOpt.acquisitions.AcquisitionEI_hmc(model, space, optimizer=aquisition_optimizer)
    
    # --- CHOOSE the intial design
    initial_design = GPyOpt.util.stats.initial_design('random', space.get_continuous_bounds(), 5)
    
    # BO object
    bo = GPyOpt.core.BO(model, space, objective, acquisition, initial_design)

                                                
    print '-----'
    print '----- Running demo. It may take a few seconds.'
    print '-----'
    
    # --- Stop conditions
    max_time  = 20 
    max_iter  = 3
    tolerance = 1e-8     # distance between two consecutive observations  

    # Run the optimization                                                  
    bo.run_optimization(max_iter = max_iter, max_time = max_time, eps = tolerance) 
                            

    # --- Plots
    if plots:
        func.plot()
        bo.plot_acquisition()
        bo.plot_convergence()
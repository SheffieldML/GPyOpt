def simple_optimization_1d(plots=True):
    import GPyOpt
    from numpy.random import seed
    seed(1234)
        
    # --- Fucntion to optimize
    func  = GPyOpt.objective_examples.experiments1d.forrester()  
    cost = lambda x: 2*x

    # --- Space design
    space = GPyOpt.Design_space([{'domain':(0,1)}])

    # --- Object of 
    objective = GPyOpt.core.task.SingleObjective(func.f, space, cost)

    # --- CHOOSE the model type
    model = GPyOpt.models.GPModel(exact_feval=True, num_hmc_samples = 500)
    
    # --- CHOOSE the acquisition optimizer
    aquisition_optimizer = GPyOpt.optimization.ContAcqOptimizer(space, 50)
    
    # --- CHOOSE the type of acquisition
    acquisition = GPyOpt.acquisitions.AcquisitionMPI(model, space, optimizer=aquisition_optimizer)
    
    # --- CHOOSE the type of acquisition
    initial_design = GPyOpt.util.stats.initial_design('random', space.get_continuous_bounds(), 5)
    
    # Create the BO object
    bo = GPyOpt.core.BO(model, space, objective, acquisition, initial_design)

    # --- Problem definition and optimization
    max_time = 20                                                         

    print '-----'
    print '----- Running demo. It may take a few seconds.'
    print '-----'
    
    # Run the optimization                                                  
    bo.run_optimization(max_time=max_time, eps=1e-8) 
                            

    # --- Plots
    if plots:
        objective_true.plot()
        #bo.plot_acquisition()
        #bo.plot_convergence()
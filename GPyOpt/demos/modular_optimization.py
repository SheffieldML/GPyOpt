def modular_optimization(plots=True):
    import GPyOpt
    from numpy.random import seed
    seed(1234)
        
    # --- Fucntion to optimize
    func  = GPyOpt.objective_examples.experiments2d.sixhumpcamel() 
    #func  = GPyOpt.objective_examples.experiments1d.forrester() 

    # --- Space design
    space = GPyOpt.Design_space(space =[{'name': 'var_1', 'type': 'continuous', 'domain': (-1,1)},
                                        {'name': 'var_2', 'type': 'continuous', 'domain': (-1.3,1.3)}])
    #space = GPyOpt.Design_space([{'type': 'continuous','domain':func.bounds[0]}])

    # --- Objective
    objective = GPyOpt.core.task.SingleObjective(func.f, space)

    # --- CHOOSE the model type
    model = GPyOpt.models.RFModel()
    
    # --- CHOOSE the acquisition optimizer
    aquisition_optimizer = GPyOpt.optimization.ContAcqOptimizer(space, 1000, search=True)
    
    # --- CHOOSE the type of acquisition
    acquisition = GPyOpt.acquisitions.AcquisitionEI(model, space, optimizer=aquisition_optimizer)
    
    # --- CHOOSE the intial design
    initial_design = GPyOpt.util.stats.initial_design('random', space.get_continuous_bounds(), 10)
    
    # BO object
    bo = GPyOpt.core.BO(model, space, objective, acquisition, initial_design)
                                                
    print '-----'
    print '----- Running demo. It may take a few seconds.'
    print '-----'
    
    # --- Stop conditions
    max_time  = None 
    max_iter  = 3
    tolerance = 1e-8     # distance between two consecutive observations  

    # Run the optimization                                                  
    bo.run_optimization(max_iter = max_iter, max_time = max_time, eps = tolerance, verbose=False) 
                        

    # --- Plots
    if plots:
        func.plot()
        bo.plot_acquisition()
        #bo.plot_convergence()
    return bo


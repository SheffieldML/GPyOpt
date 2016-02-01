import GPyOpt
import numpy as np
import os

def run_eval(f_obj, f_bounds, f_inits, method_config, name = 'run', outpath='.', time_limit=-1, unittest = False):
    m_c = method_config

    # Inital values
    xs_init = f_inits[0]
    ys_init = f_obj(xs_init)

    # Create the Bayesian Optimization Object
    bo = GPyOpt.methods.BayesianOptimization(f_obj, 
                                            bounds                  = f_bounds, 
                                            kernel                  = m_c['kernel'],
                                            X                       = xs_init.copy(), 
                                            Y                       = ys_init.copy(), 
                                            numdata_initial_design  = m_c['numdata_initial_design'],
                                            type_initial_design     = m_c['type_initial_design'],
                                            model_optimize_interval = m_c['model_optimize_interval'], 
                                            acquisition             = m_c['acquisition_name'], 
                                            acquisition_par         = m_c['acquisition_par'],
                                            model_optimize_restarts = m_c['model_optimize_restarts'], 
                                            sparseGP                = m_c['sparseGP'], 
                                            num_inducing            = m_c['num_inducing'], 
                                            normalize               = m_c['normalize'],
                                            exact_feval             = m_c['exact_feval'],
                                            verbosity               = m_c['verbosity'])

    # Run the optimization
    bo.run_optimization(max_iter                = m_c['max_iter'], 
                        batch_method            = m_c['batch_method'], 
                        n_inbatch               = m_c['n_inbatch'], 
                        acqu_optimize_method    = m_c['acqu_optimize_method'],
                        acqu_optimize_restarts  = m_c['acqu_optimize_restarts'], 
                        n_procs                 = m_c['n_procs'],
                        eps                     = m_c['eps'],
                        true_gradients          = m_c['true_gradients'],
                        verbose                 = m_c['verbosity'])

    # Save value of X in a file
    results = bo.X
    
    # Used to run the original resuult used for testing
    if unittest==False:
        np.savetxt(os.path.join(outpath,name+'.txt'),results)

    return results

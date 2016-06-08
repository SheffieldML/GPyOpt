# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import GPyOpt
import numpy as np
import os



def run_eval(problem_config, f_inits, method_config, name = 'run', outpath='.', time_limit=-1, unittest = False):
    """
    This is a driver for running the optimization in the unittests.
    """

    # Methods configuration
    m_c = method_config

    # Inital values
    xs_init = f_inits[0]
    f_obj   = problem_config['objective'] 
    ys_init = f_obj(xs_init)

    # Create the Bayesian Optimization Object
    bo = GPyOpt.methods.BayesianOptimization(   f                           = problem_config['objective'], 
                                                domain                      = problem_config['domain'], 
                                                constrains                  = problem_config['constrains'], 
                                                cost_withGradients          = problem_config['cost_withGradients'], 
                                                model_type                  = m_c['model_type'], 
                                                X                           = xs_init.copy(), 
                                                Y                           = ys_init.copy(), 
                                                initial_design_numdata      = m_c['initial_design_type'], 
                                                initial_design_type         = m_c['initial_design_type'], 
                                                acquisition_type            = m_c['acquisition_type'], 
                                                normalize_Y                 = m_c['normalize_Y'], 
                                                exact_feval                 = m_c['exact_feval'], 
                                                acquisition_optimizer_type  = m_c['acquisition_optimizer_type'], 
                                                model_update_interval       = m_c['model_update_interval'], 
                                                verbosity                   = m_c['verbosity'], 
                                                evaluator_type              = m_c['evaluator_type'], 
                                                batch_size                  = m_c['batch_size'], 
                                                num_cores                   = m_c['num_cores'])

    # Run the optimization
    bo.run_optimization(max_iter        = m_c['max_iter'], 
                        max_time        = m_c['max_time'],
                        eps             = m_c['eps'],
                        verbosity       = m_c['verbosity'])

    # Save value of X in a file
    results = bo.X
    
    # Used to run the original resuult used for testing
    if unittest==False:
        np.savetxt(os.path.join(outpath,name+'.txt'),results)
        print('*********************************************************************************')
        print('This is not a test. This option is used to generate the files used for the tests.')
        print('*********************************************************************************')

    return results

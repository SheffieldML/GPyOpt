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

    # Initial values
    xs_init = f_inits[:]
    f_obj   = problem_config['objective']
    ys_init = f_obj(xs_init)

    # Create the Bayesian Optimization Object
    bo = GPyOpt.methods.BayesianOptimization(   f                           = problem_config['objective'],
                                                domain                      = problem_config['domain'],
                                                constraints                 = problem_config['constraints'],
                                                cost_withGradients          = problem_config['cost_withGradients'],
                                                model_type                  = m_c['model_type'],
                                                X                           = xs_init.copy(),
                                                Y                           = ys_init.copy(),
                                                initial_design_numdata      = m_c['initial_design_numdata'],
                                                initial_design_type         = m_c['initial_design_type'],
                                                acquisition_type            = m_c['acquisition_type'],
                                                normalize_Y                 = m_c['normalize_Y'],
                                                exact_feval                 = m_c['exact_feval'],
                                                acquisition_optimizer_type  = m_c['acquisition_optimizer_type'],
                                                model_update_interval       = m_c['model_update_interval'],
                                                verbosity                   = m_c['verbosity'],
                                                evaluator_type              = m_c['evaluator_type'],
                                                batch_size                  = m_c['batch_size'],
                                                num_cores                   = m_c['num_cores'],
                                                de_duplication              = m_c.get('de_duplication',False)
                                                )

    if 'context' not in problem_config.keys():
        problem_config['context'] = None

    # Run the optimization
    bo.run_optimization(max_iter        = m_c['max_iter'],
                        max_time        = m_c['max_time'],
                        eps             = m_c['eps'],
                        verbosity       = m_c['verbosity'],
                        context         = problem_config['context'])

    # Save value of X in a file
    results = bo.X

    # Used to run the original resuult used for testing
    if unittest==False:
        np.savetxt(os.path.join(outpath,name+'.txt'),results)
        print('*********************************************************************************')
        print('This is not a test. This option is used to generate the files used for the tests.')
        print('*********************************************************************************')

    return results

def run_evaluation_in_steps(problem_config, f_inits, method_config, num_steps=5):
    """
    This is a driver for running the optimization in the unittests.
    It executes optimization for a set number of steps.
    """

    # Methods configuration
    m_c = method_config

    # Initial values
    xs_init = f_inits[:]
    f_obj   = problem_config['objective']
    ys_init = f_obj(xs_init)

    X = xs_init
    Y = ys_init
    for step in range(num_steps):
        bo = GPyOpt.methods.BayesianOptimization(
                f                           = None,
                domain                      = problem_config['domain'],
                constraints                 = problem_config['constraints'],
                cost_withGradients          = problem_config['cost_withGradients'],
                model_type                  = m_c['model_type'],
                X                           = X,
                Y                           = Y,
                initial_design_numdata      = m_c['initial_design_numdata'],
                initial_design_type         = m_c['initial_design_type'],
                acquisition_type            = m_c['acquisition_type'],
                normalize_Y                 = m_c['normalize_Y'],
                exact_feval                 = m_c['exact_feval'],
                acquisition_optimizer_type  = m_c['acquisition_optimizer_type'],
                model_update_interval       = m_c['model_update_interval'],
                verbosity                   = m_c['verbosity'],
                evaluator_type              = m_c['evaluator_type'],
                batch_size                  = m_c['batch_size'],
                num_cores                   = m_c['num_cores'],
                de_duplication              = m_c.get('de_duplication',False))

        if 'context' not in problem_config.keys():
            problem_config['context'] = None

        next_locations = bo.suggest_next_locations()
        X = np.vstack((X, next_locations))
        Y = f_obj(X)

    return X



import GPyOpt
import time
import os
import numpy as np


def run_eval(f_obj, f_bounds, f_inits, method_config, name, outpath='.', time_limit=-1, unittest = False):
    m_c = method_config

    # Inital values
    xs_init = f_inits[0]
    ys_init = f_obj(xs_init)

    # Create the Bayesian Optimization Object
    bo = GPyOpt.methods.BayesianOptimization(f_obj, bounds=f_bounds, X= xs_init.copy(), Y=ys_init.copy(), acquisition=m_c['acquisition_name'], acquisition_par = m_c['acquisition_par'],normalize=True)
    
    # Run the optimization
    bo.run_optimization(max_iter = m_c['max_iter'], batch_method = m_c['batch_method'], n_inbatch= m_c['n_inbatch'], acqu_optimize_method=m_c['acqu_optimize_method'],acqu_optimize_restarts= m_c['acqu_optimize_restarts'], n_procs=m_c['n_procs'],verbose=False)

    # Save value of X in a file
    results = bo.X
    
    if unittest==False:
        np.savetxt(os.path.join(outpath,name+'.txt'),results)

    return results

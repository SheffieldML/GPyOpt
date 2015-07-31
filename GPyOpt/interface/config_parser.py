
import sys
import optparse
import os
import numpy as np
import json


default_config = {
    "language"        : "PYTHON",
    "likelihood"      : "NOISELESS",

    "model": {
        "type" : "GP",
        "inducing_points": 10,
        "initial-points": 5,
        "design-inital-points": "latin",
        "normalized-evaluations": False,
        "optimization-restarts": 1,
        "optimization-interval": 1
        },

    "resources": {
        "iterations" :  100,
        "running_time":  60,
        "cores": 1  
        },

    "acquisition": {
        "type" : 'EI',
        "parameter": 0,
        "true-gradients": True,
        "optimization-method": "DIRECT",
        "optimization-restarts": 10
        },

    "parallelization":{
        "distributed": False,
        "type":"lp",
        "batch-size":1,
        },

    "output":{
        "vebosity": False,
        "file-report": True,
        "convergence-plot": False
        },
}



def update_config(config_new, config_default):

	if any([isinstance(v, dict) for v in config_new.values()]):
		for k,v in config_new.iteritems():
			if isinstance(v,dict) and k in config_default:
				update_config(config_new[k],config_default[k])
			else:
				config_default[k] = v
	else:
		config_default.update(config_new)
	return config_default


def parser(input_file_path='config.json'):


    # --- Read .json file
    try:
        with open(input_file_path, 'r') as config_file:
            config_new = json.load(config_file)
    except:
        raise Exception('File config.json not loaded properly. Please check it an try again.')

    import copy
    options = update_config(config_new, copy.deepcopy(default_config))

    return options
        
        



"""
f,#---- DONE
bounds=None, f,#---- DONE
kernel=None, f,#---- DONE
X=None, f,#---- DONE
Y=None, f,#---- DONE
numdata_inital_design = None, f,#---- DONE
type_initial_design='random', f,#---- DONE
model_optimize_interval=1, f,#---- DONE
acquisition='EI',f,#---- DONE
acquisition_par= 0.00, f,#---- DONE
model_optimize_restarts=10, f,#---- DONE
sparseGP=False, f,#---- DONE
num_inducing=None, f,#---- DONE
normalize=False, #---- DONE
true_gradients=True,#---- DONE
exact_feval=False, #---- DONE
verbosity=0):



max_iter = None,#---- DONE
n_inbatch=1, #---- DONE
acqu_optimize_method='fast_random', #---- DONE
acqu_optimize_restarts=200, #---- DONE
batch_method='predictive',#---- DONE
"""
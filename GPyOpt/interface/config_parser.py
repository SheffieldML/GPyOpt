import sys
import optparse
import os
import numpy as np
import json

default_config = {
    "language"        : "PYTHON",
    "experiment-name" : "no-named-experiment",
    'support-multi-eval': False,
    "likelihood"      : "NOISELESS",

    "model": {
        "type" : "GP",
        "inducing-points": 10,
        "initial-points": 10,
        "design-initial-points": "random",
        "normalized-evaluations": True,
        "optimization-restarts": 5,
        "optimization-interval": 1,
        },

    "resources": {
        "maximum-iterations" :  20,
        "max-run-time": 'NA', #minutes
        "cores": 1,
        "tolerance": 1e-8,
        'iterations_per_call': 1,
        },

    "acquisition": {
        "type" : 'EI',
        "parameter": 0,
        "true-gradients": True,
        "optimization-method": "fast_random",
        "optimization-restarts": 200
        },

    "parallelization":{
        "distributed": False,
        "type":"lp",
        "batch-size":1,
        },

    "output":{
        "verbosity": False,
        "file-report": {
                'type': 'report',
                'filename': None,
                'interval': -1,
        },
        "Ybest": {
                'type': 'logger',
                'content': 'ybest',
                'format': 'csv',
                'filename': None,
                'interval': 1,
        },
#        "convergence-plot": False
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
            config_file.close()
    except:
        raise Exception('File config.json not loaded properly. Please check it an try again.')

    import copy
    options = update_config(config_new, copy.deepcopy(default_config))

    return options
        

# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import sys
import optparse
import os
import numpy as np
import json

default_config = {
    "language"        : "PYTHON",
    "experiment-name" : "no-named-experiment",
    "likelihood"      : "NOISELESS",
    'initialization': {
            'type':'random',
            'num-eval':5,
        },

    "model": {
        "type" : "GP",
        "num_inducing": 10,
        },
                  
    "constraints": [],

    "resources": {
        "maximum-iterations" :  20,
        "max-run-time": 'NA', #minutes
        "cores": 1,
        "tolerance": 1e-8,
        },

    "acquisition": {
        "type" : 'EI',
        "jitter" : 0.01,
        "optimizer" : {
                "name": "lbfgs"
            },
        "evaluator" : {
                "type" : "sequential"
            }
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
        },
}


def update_config(config_new, config_default):

    '''
    Updates the loaded method configuration with default values.
    '''
    if any([isinstance(v, dict) for v in list(config_new.values())]):
        for k,v in list(config_new.items()):
            if isinstance(v,dict) and k in config_default:
                update_config(config_new[k],config_default[k])
            else:
                config_default[k] = v
    else:
        config_default.update(config_new)
    return config_default


def parser(input_file_path='config.json'):
    '''
    Parser for the .json file containing the configuration of the method.
    '''

    # --- Read .json file
    try:
        with open(input_file_path, 'r') as config_file:
            config_new = json.load(config_file)
            config_file.close()
    except:
        raise Exception('Config file "'+input_file_path+'" not loaded properly. Please check it an try again.')

    import copy
    options = update_config(config_new, copy.deepcopy(default_config))

    return options
        

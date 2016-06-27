# Copyright (c) 2014, GPyOpt authors (see AUTHORS.txt).
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import os
import numpy as np

def load_objective(config):
    """
    Loads the objective function from a .json file.
    """

    assert 'prjpath' in config
    assert 'main-file' in config, "The problem file ('main-file') is missing!"
    
    os.chdir(config['prjpath'])
    if config['language'].lower()=='python':
        assert config['main-file'].endswith('.py'), 'The python problem file has to end with .py!'
        import imp
        m = imp.load_source(config['main-file'][:-3], os.path.join(config['prjpath'],config['main-file']))
        func = m.__dict__[config['main-file'][:-3]]
    return func


# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from .base import BOModel
from .gpmodel import GPModel, GPModel_MCMC
from .rfmodel import RFModel
from .warpedgpmodel import WarpedGPModel
from .input_warped_gpmodel import InputWarpedGPModel
#from . import gpykernels

def select_model(name):
    if name == 'GP':
        return GPModel
    elif name == 'GP_MCMC':
        return GPModel_MCMC
    elif name == 'RF':
        return RFModel
    elif name == 'warpGP':
        return WarpedGPModel
    else:
        raise Exception('Invalid model selected.')

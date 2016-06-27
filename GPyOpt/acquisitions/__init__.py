# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from .base import AcquisitionBase
from .EI import AcquisitionEI
from GPyOpt.acquisitions.EI_mcmc import AcquisitionEI_MCMC
from .MPI import AcquisitionMPI
from .MPI_mcmc import AcquisitionMPI_MCMC
from .LCB import AcquisitionLCB
from .LCB_mcmc import AcquisitionLCB_MCMC
from .LP import AcquisitionLP

def select_acquisition(name):
    '''
    Acquisition selector
    '''
    if name == 'EI':
        return AcquisitionEI
    elif name == 'EI_MCMC':
        return AcquisitionEI_MCMC
    elif name == 'LCB':
        return AcquisitionLCB
    elif name == 'LCB_MCMC':
        return AcquisitionLCB_MCMC
    elif name == 'MPI':
        return AcquisitionMPI
    elif name == 'MPI_MCMC':
        return AcquisitionMPI_MCMC
    elif name == 'LP':
        return AcquisitionLP
    else:
        raise Exception('Invalid acquisition selected.')

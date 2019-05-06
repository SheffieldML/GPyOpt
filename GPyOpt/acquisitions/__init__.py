# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from .base import AcquisitionBase
from .EI import AcquisitionEI
from .EI_mcmc import AcquisitionEI_MCMC
from .EI_PoF import AcquisitionEI_PoF
from .MPI import AcquisitionMPI
from .MPI_mcmc import AcquisitionMPI_MCMC
from .MPI_PoF import AcquisitionMPI_PoF
from .LCB import AcquisitionLCB
from .LCB_mcmc import AcquisitionLCB_MCMC
from .LCB_PoF import AcquisitionLCB_PoF
from .LP import AcquisitionLP
from .ES import AcquisitionEntropySearch

def select_acquisition(name):
    '''
    Acquisition selector
    '''
    if name == 'EI':
        return AcquisitionEI
    elif name == 'EI_MCMC':
        return AcquisitionEI_MCMC
    elif name == 'EI_PoF':
        return AcquisitionEI_PoF
    elif name == 'LCB':
        return AcquisitionLCB
    elif name == 'LCB_MCMC':
        return AcquisitionLCB_MCMC
    elif name == 'LCB_PoF':
        return AcquisitionLCB_PoF
    elif name == 'MPI':
        return AcquisitionMPI
    elif name == 'MPI_MCMC':
        return AcquisitionMPI_MCMC
    elif name == 'MPI_PoF':
        return AcquisitionMPI_PoF
    elif name == 'LP':
        return AcquisitionLP
    elif name == 'ES':
        return AcquisitionEntropySearch
    else:
        raise Exception('Invalid acquisition selected.')

# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from GPyOpt.core.task.space import Design_space
from . import core
from . import methods
from . import util
from . import plotting
from . import interface
from . import models
from . import acquisitions
from . import optimization
from . import objective_examples
from . import objective_examples as fmodels

from .__version__ import __version__

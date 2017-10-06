# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from .base import EvaluatorBase
from .sequential import Sequential
from .batch_random import RandomBatch
from .batch_local_penalization import LocalPenalization
from .batch_thompson import ThompsonBatch

def select_evaluator(name):
    if name == 'sequential':
        return Sequential
    elif name == 'random':
        return RandomBatch
    elif name == 'local_penalization':
        return LocalPenalization
    elif name == 'thompson_sampling':
        return ThompsonBatch
    else:
        raise Exception('Invalid acquisition evaluator selected.')

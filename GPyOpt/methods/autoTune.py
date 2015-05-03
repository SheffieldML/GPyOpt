# Copyright (c) 2015, Javier Gonzalez
# Copyright (c) 2015, the GPy Authors (see GPy AUTHORS.txt)
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from .bayesian_optimzation import BayesianOptimization

class autoTune(BayesianOptimization):
    def __init__(self, f, bounds=None, max_iter=None, stop_criteria = 1e-16, save_file = None, plot_file =None):
        self.f = f
        self.bounds = bounds
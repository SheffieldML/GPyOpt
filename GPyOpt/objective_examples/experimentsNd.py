# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np


class alpine1:
    '''
    Alpine1 function
    '''
    def __init__(self, input_dim, bounds=None, sd=0.):
        if bounds is None:
            self.bounds = [(-10., 10.)] * input_dim
        else:
            self.bounds = bounds

        self.min = np.array([[0] * input_dim])
        self.fmin = 0
        self.input_dim = input_dim
        self.sd = sd
        return

    def f(self, X):
        assert X.shape[1] == self.input_dim,\
            'Wrong input dimension! {0}'.format(X.shape)
        fval = (X * np.sin(X) + 0.1 * X).sum(axis=1)
        noise = np.random.normal(0, self.sd, size=fval.shape)
        return fval + noise


class alpine2:
    '''
    Alpine2 function
    '''
    def __init__(self, input_dim, bounds=None, sd=0.):
        if bounds is None:
            self.bounds = [(1, 10)] * input_dim
        else:
            self.bounds = bounds

        self.min = np.array([[7.917] * input_dim])
        self.fmin = -(2.808**input_dim)
        self.input_dim = input_dim
        self.sd = sd
        return

    def f(self, X):
        assert X.shape[1] == self.input_dim,\
            'Wrong input dimension! {0}'.format(X.shape)
        fval = (np.cumprod(np.sqrt(X), axis=1)[:, self.input_dim - 1] *
                np.cumprod(np.sin(X), axis=1)[:, self.input_dim - 1])
        noise = np.random.normal(0, self.sd, size=fval.shape)
        return -fval + noise


class ackley:
    '''
    Ackley function (https://www.sfu.ca/~ssurjano/ackley.html)
    '''
    def __init__(self, input_dim, bounds=None, sd=0.):
        '''
        :param sd: standard deviation, for noisy evaluations of the function.
        '''
        self.input_dim = input_dim
        if bounds is None:
            self.bounds = [(-32.768, 32.768)] * self.input_dim
        else:
            self.bounds = bounds

        self.min = np.array([[0.] * self.input_dim])
        self.fmin = 0
        self.sd = sd
        return

    def f(self, X):
        assert X.shape[1] == self.input_dim,\
            'Wrong input dimension! {0}'.format(X.shape)
        a, b, c = 20, 0.2, 2 * np.pi
        fval = (-a * np.exp((-b / np.sqrt(self.input_dim)) *
                            np.sqrt(np.linalg.norm(X, axis=1))) -
                np.exp((1. / self.input_dim) * np.sum(np.cos(c * X), axis=1)) +
                + a + np.e)
        noise = np.random.normal(0, self.sd, size=fval.shape)
        return fval + noise

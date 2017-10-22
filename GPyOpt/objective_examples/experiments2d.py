# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

try:
    import matplotlib.pyplot as plt
except:
    pass

import numpy as np


class function2d:
    '''
    This is a benchmark of bi-dimensional functions interesting to optimize.
    '''
    def __init__(self):
        self.input_dim = 2
        return

    def plot(self, ax=None, plt_type='contourf'):
        bounds = self.bounds
        x1 = np.arange(bounds[0][0], bounds[0][1], 0.01)
        x2 = np.arange(bounds[1][0], bounds[1][1], 0.01)
        n_grid = len(x1)
        X1, X2 = np.meshgrid(x1, x2)
        X = np.hstack((X1.reshape(n_grid**2, 1), X2.reshape(n_grid**2, 1)))
        Y = self.f(X)

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            call_show = True
        else:
            call_show = False

        if plt_type.lower() == 'contourf':
            im = ax.contourf(X1, X2, Y.reshape((n_grid, n_grid)), label='f(x)')
            ax.figure.colorbar(im, ax=ax)
        elif plt_type.lower() == 'contour':
            ax.contour(X1, X2, Y.reshape((n_grid, n_grid)), label='f(x)')
        else:
            raise NotImplementedError("No such plot_type, options are"
                                      "'contourf' or 'contour'")

        if (len(self.min) > 1):
            ax.plot(
                np.array(self.min)[:, 0],
                np.array(self.min)[:, 1],
                color='k',
                linestyle='',
                marker='X',
                label=u'Minima')
        else:
            ax.plot(
                np.array(self.min[0][0]),
                np.array(self.min[0][1]),
                color='k',
                linestyle='',
                marker='X',
                label=u'Minimum')
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.set_xlim(x1[0], x1[-1])
        ax.set_ylim(x2[0], x2[-1])
        ax.legend()
        if ax.title.get_text == '':
            ax.set_title(self.name)
        if call_show:
            plt.show()
        return


class rosenbrock(function2d):
    '''Rosenbrock function (https://www.sfu.ca/~ssurjano/rosen.html)

    :param bounds: the box constraints to define the domain in which
      the function is optimized.
    :param sd: standard deviation, to
      generate noisy evaluations of the function.
    '''
    def __init__(self, bounds=None, sd=0):
        super(rosenbrock, self).__init__()

        if bounds is None:
            self.bounds = [(-1.5, 2), (-0.5, 3)]
        else:
            self.bounds = bounds

        self.min = np.array([[1, 1]])
        self.fmin = 0
        self.name = 'Rosenbrock'
        self.sd = sd
        return

    def f(self, X):
        assert X.shape[1] == self.input_dim, 'Wrong dimension! X.shape = '\
            '{0}'.format(X.shape)

        x1 = X[:, 0]
        x2 = X[:, 1]
        fval = 100 * (x2 - x1 ** 2) ** 2 + (1 - x1) ** 2
        noise = np.random.normal(0, self.sd, size=fval.shape)
        return fval + noise


class beale(function2d):
    '''
    beale function (https://www.sfu.ca/~ssurjano/beale.html)
    '''
    def __init__(self, bounds=None, sd=0):
        super(beale, self).__init__()

        if bounds is None:
            self.bounds = [(-4.5, 4.5), (-4.5, 4.5)]
        else:
            self.bounds = bounds

        self.min = np.array([[3, 0.5]])
        self.fmin = 0
        self.name = 'Beale'
        self.sd = sd
        return

    def f(self, X):
        x1 = X[:, 0]
        x2 = X[:, 1]
        fval = (1.5 - x1 + x1 * x2) ** 2 +\
               (2.25 - x1 + x1 * (x2 ** 2)) ** 2 +\
               (2.625 - x1 + x1 * (x2 ** 3)) ** 2
        noise = np.random.normal(0, self.sd, size=fval.shape)
        return fval + noise


class dropwave(function2d):
    '''
    dropwave function (https://www.sfu.ca/~ssurjano/drop.html)
    '''

    def __init__(self, bounds=None, sd=0):
        super(dropwave, self).__init__()
        if bounds is None:
            self.bounds = [(-2, 2), (-2, 2)]
        else:
            self.bounds = bounds

        self.min = np.array([[0, 0]])
        self.fmin = -1
        self.name = 'dropwave'
        self.sd = sd
        return

    def f(self, X):
        assert X.shape[1] == self.input_dim, 'Wrong dimension! X.shape = '\
            '{0}'.format(X.shape)

        x1 = X[:, 0]
        x2 = X[:, 1]
        fval = -(1 + np.cos(12 * np.sqrt(x1 ** 2 + x2 ** 2))) / (
            0.5 * (x1 ** 2 + x2 ** 2) + 2)
        noise = np.random.normal(0, self.sd, size=fval.shape)
        return fval + noise


class cosines(function2d):
    '''
    Cosines function
    '''
    def __init__(self, bounds=None, sd=0):
        super(cosines, self).__init__()
        if bounds is None:
            self.bounds = [(0, 1), (0, 1)]
        else:
            self.bounds = bounds

        self.min = np.array([[0.31426205, 0.30249864]])
        self.fmin = 1.59622468
        self.name = 'Cosines'
        self.sd = sd
        return

    def f(self, X):
        assert X.shape[1] == self.input_dim, 'Wrong dimension! X.shape = '\
            '{0}'.format(X.shape)

        x1 = X[:, 0]
        x2 = X[:, 1]
        u = 1.6 * x1 - 0.5
        v = 1.6 * x2 - 0.5
        fval = 1 - (u ** 2 + v ** 2 - 0.3 * np.cos(3 * np.pi * u) -
                    0.3 * np.cos(3 * np.pi * v))
        noise = np.random.normal(0, self.sd, size=fval.shape)
        return fval + noise


class branin(function2d):
    '''
    Branin function (https://www.sfu.ca/~ssurjano/branin.html)
    '''

    def __init__(self, bounds=None, sd=0):
        super(branin, self).__init__()
        if bounds is None:
            self.bounds = [(-5, 10), (1, 15)]
        else:
            self.bounds = bounds

        self.min = np.array([[-np.pi, 12.275],
                            [np.pi, 2.275],
                            [9.42478, 2.475]])
        self.fmin = 0.397887
        self.name = 'Branin'
        self.sd = sd

    def f(self, X):
        assert X.shape[1] == self.input_dim, 'Wrong dimension! X.shape = '\
            '{0}'.format(X.shape)

        a = 1.
        b = 5.1 / (4 * np.pi ** 2)
        c = 5 / np.pi
        r = 6.
        s = 10.
        t = 1. / (8 * np.pi)

        x1 = X[:, 0]
        x2 = X[:, 1]
        fval = (a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 +
                s * (1 - t) * np.cos(x1) + s)
        noise = np.random.normal(0, self.sd, size=fval.shape)
        return fval + noise


class goldstein(function2d):
    '''
    Goldstein function
    '''

    def __init__(self, bounds=None, sd=0):
        super(goldstein, self).__init__()
        if bounds is None:
            self.bounds = [(-2, 2), (-2, 2)]
        else:
            self.bounds = bounds

        self.min = np.array([[0, -1]])
        self.fmin = 3
        self.name = 'Goldstein'
        self.sd = sd

    def f(self, X):
        assert X.shape[1] == self.input_dim, 'Wrong dimension! X.shape = '\
            '{0}'.format(X.shape)

        x1 = X[:, 0]
        x2 = X[:, 1]
        fact1a = (x1 + x2 + 1)**2
        fact1b = (19 - 14 * x1 + 3 * x1**2 - 14 * x2 +
                  6 * x1 * x2 + 3 * x2**2)
        fact1 = 1 + fact1a * fact1b

        fact2a = (2 * x1 - 3 * x2)**2
        fact2b = (18 - 32 * x1 + 12 * x1**2 +
                  48 * x2 - 36 * x1 * x2 + 27 * x2**2)
        fact2 = 30 + fact2a * fact2b

        fval = fact1 * fact2
        noise = np.random.normal(0, self.sd, size=fval.shape)
        return fval + noise


class sixhumpcamel(function2d):
    '''
    Six hump camel function (https://www.sfu.ca/~ssurjano/camel6.html)
    '''

    def __init__(self, bounds=None, sd=0):
        super(sixhumpcamel, self).__init__()
        if bounds is None:
            self.bounds = [(-2, 2), (-1, 1)]
        else:
            self.bounds = bounds

        self.min = np.array([[0.0898, -0.7126],
                             [-0.0898, 0.7126]])
        self.fmin = -1.0316
        self.name = 'Six-hump camel'
        self.sd = sd
        return

    def f(self, X):
        assert X.shape[1] == self.input_dim, 'Wrong dimension! X.shape = '\
            '{0}'.format(X.shape)

        x1 = X[:, 0]
        x2 = X[:, 1]
        term1 = (4 - 2.1 * x1**2 + (x1**4) / 3) * x1**2
        term2 = x1 * x2
        term3 = (-4 + 4 * x2**2) * x2**2
        fval = term1 + term2 + term3
        noise = np.random.normal(0, self.sd, size=fval.shape)
        return fval + noise


class mccormick(function2d):
    '''
    Mccormick function (https://www.sfu.ca/~ssurjano/mccorm.html)
    '''
    def __init__(self, bounds=None, sd=0):
        super(mccormick, self).__init__()
        if bounds is None:
            self.bounds = [(-1.5, 4), (-3, 4)]
        else:
            self.bounds = bounds

        self.min = np.array([[-0.54719, -1.54719]])
        self.fmin = -1.9133
        self.name = 'Mccormick'
        self.sd = sd
        return

    def f(self, X):
        assert X.shape[1] == self.input_dim, 'Wrong dimension! X.shape = '\
            '{0}'.format(X.shape)

        x1 = X[:, 0]
        x2 = X[:, 1]
        term1 = np.sin(x1 + x2)
        term2 = (x1 - x2)**2
        term3 = -1.5 * x1
        term4 = 2.5 * x2
        fval = term1 + term2 + term3 + term4 + 1
        noise = np.random.normal(0, self.sd, size=fval.shape)
        return fval + noise


class powers(function2d):
    '''
    Powers function
    '''
    def __init__(self, bounds=None, sd=0):
        super(powers, self).__init__()
        if bounds is None:
            self.bounds = [(-1, 1), (-1, 1)]
        else:
            self.bounds = bounds
        self.min = np.array([[0, 0]])
        self.fmin = 0
        self.name = 'Sum of Powers'
        self.sd = sd
        return

    def f(self, X):
        assert X.shape[1] == self.input_dim, 'Wrong dimension! X.shape = '\
            '{0}'.format(X.shape)

        x1 = X[:, 0]
        x2 = X[:, 1]
        fval = abs(x1)**2 + abs(x2)**3
        noise = np.random.normal(0, self.sd, size=fval.shape)
        return fval + noise


class eggholder(function2d):
    def __init__(self, bounds=None, sd=0):
        super(eggholder, self).__init__()
        if bounds is None:
            self.bounds = [(-5.12, 5.12), (-5.12, 5.12)]
        else:
            self.bounds = bounds

        self.min = np.array([[512, 404.2319]])
        self.fmin = -959.6407
        self.name = 'Egg-holder'
        self.sd = sd

    def f(self, X):
        assert X.shape[1] == self.input_dim, 'Wrong dimension! X.shape = '\
            '{0}'.format(X.shape)

        x1 = X[:, 0]
        x2 = X[:, 1]
        fval = -(x2 + 47) * np.sin(np.sqrt(
            abs(x2 + x1 / 2 + 47))) + -x1 * np.sin(
                np.sqrt(abs(x1 - (x2 + 47))))
        noise = np.random.normal(0, self.sd, size=fval.shape)
        return fval + noise

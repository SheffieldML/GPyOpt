# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

import numpy as np
import matplotlib.pyplot as plt


def _normalize_acqu(acqu):
    acqu_normalized = max(acqu) - acqu
    return acqu_normalized / max(acqu_normalized)


def plot_acquisition(bounds, input_dim, model, Xdata, Ydata,
                     acquisition_function, suggested_sample,
                     filename=None, fig=None):
    '''
    Plots of the model and the acquisition function in 1D and 2D examples.
    '''
    if fig is None:
        # Whether or not to call plt.show() at the end.  plt.show() will
        # still not be called if a filename is supplied.
        call_show = True
        fig = plt.figure()
    else:
        call_show = False

    # Plots in dimension 1
    if input_dim == 1:
        ax = fig.add_subplot(1, 1, 1)
        x_grid = np.arange(bounds[0][0], bounds[0][1], 0.001)

        # Need column vectors for acquisition_function and predict
        x_grid = x_grid[:, np.newaxis]
        acqu = _normalize_acqu(acquisition_function(x_grid))

        # acqu_normalized = (-acqu - min(-acqu)) / (max(-acqu - min(-acqu)))
        m, v = model.predict(x_grid)  # Mean and variance from GPy model
        x_grid, m, v = x_grid[:, 0], m[:, 0], v[:, 0]  # Convert to 1D arrays
        std = np.sqrt(v)
        pct95 = 1.96 * std  # 95th percentile distance from mean range

        # Plot the mean and 95pct confidence intervals of the GP
        ax.plot(x_grid, m, 'k-', lw=1, alpha=0.6)
        ax.fill_between(x_grid, m - pct95, m + pct95, color='k', alpha=0.2)

        # The points that were evaluated
        ax.plot(Xdata, Ydata, 'r.', markersize=10)

        # Vertical line at next point to evaluate
        ax.axvline(x=suggested_sample[-1], color='r')

        # Plot the acquisition function
        factor = max(m + pct95) - min(m - pct95)  # Max spread
        ax.plot(x_grid, 0.2 * factor * acqu -
                abs(min(m - pct95)) - 0.25 * factor,
                'r-', lw=2, label='Acquisition (arbitrary units)')
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.set_ylim(min(m - pct95) - 0.25 * factor,
                    max(m + pct95) + 0.05 * factor)
        ax.axvline(x=suggested_sample[-1], color='r')
        ax.legend()

        if filename is not None:
            ax.figure.savefig(filename)
        if call_show:
            plt.show()
        return

    elif input_dim == 2:
        n_grid = 200
        x = np.linspace(bounds[0][0], bounds[0][1], n_grid)
        y = np.linspace(bounds[1][0], bounds[1][1], n_grid)
        xx, yy = np.meshgrid(x, y)
        X = np.hstack((xx.flatten()[:, np.newaxis],
                       yy.flatten()[:, np.newaxis]))
        acqu = _normalize_acqu(acquisition_function(X))
        m, v = model.predict(X)
        std = np.sqrt(v)  # Standard deviation

        ax1 = fig.add_subplot(1, 3, 1)
        ax2 = fig.add_subplot(1, 3, 2)
        ax3 = fig.add_subplot(1, 3, 3)

        # We could add colorbars to each of the subplots as follows:
        # im = ax1.contourf(...)
        # fig.colorbar(im, ax=ax1)
        # But it makes the whole thing a little busy

        # ---------- Posterior mean (axis 1) ----------------
        ax1.set_title('Posterior mean')

        ax1.contourf(x, y, m.reshape(n_grid, n_grid), 100,
                     cmap='viridis')
        ax1.plot(Xdata[:, 0], Xdata[:, 1], 'r.',
                 markersize=10, label=u'Observations')

        ax1.axis('auto')

        # ---------- Posterior std (axis 1) ----------------
        ax2.set_title('Posterior std')

        ax2.plot(Xdata[:, 0], Xdata[:, 1], 'r.',
                 markersize=10, label=u'Observations')
        ax2.contourf(x, y, std.reshape(n_grid, n_grid), 100,
                     cmap='viridis')

        ax2.axis('auto')

        # ---------- Acquisition Function (axis 1) ----------------
        ax3.set_title('Acquisition function')

        ax3.plot(suggested_sample[:, 0], suggested_sample[:, 1],
                 'rX', markersize=10, label='next sample')
        ax3.contourf(x, y, acqu.reshape(n_grid, n_grid), 100)

        ax3.legend()
        ax3.axis('auto')

        if filename is not None:
            fig.savefig(filename)
        elif call_show:
            plt.show()
        return


def plot_convergence(Xdata, best_Y, filename=None, fig=None):
    '''
    Plots to evaluate the convergence of standard Bayesian optimization
    algorithms.

    Args:
        Xdata: Historical data points at which we evaulated the function
        best_Y: Historical running optimum

    KwArgs:
        filename: Location to save resulting figure
        fig: The figure on which to plot the results.  2 axes will be created.

    Returns: None
    '''
    # Distances between consecutive x's
    aux = (Xdata[1:, :] - Xdata[:-1, :]) ** 2
    distances = np.sqrt(aux.sum(axis=1))

    if fig is None:
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)
        call_show = True
    else:
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2)

    ax1.plot(distances, '-ro')
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('d(x[n], x[n-1])')
    ax1.set_title('Distance between consecutive x\'s')
    ax1.grid(True)

    # Estimated f(x) at the proposed sampling points
    ax2.plot(best_Y, '-o')
    ax2.set_title('Value of the best selected sample')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('f(x)')
    ax1.grid(True)

    if filename is not None:
        fig.savefig(filename)
    elif call_show:
        plt.show()
    return

from matplotlib import pyplot as plt
import numpy as np
import logging

from bo_configurations import *
from matplotlib import rcParams
from matplotlib.patches import Rectangle

from scipy.stats import norm

rcParams["font.size"] = 36
rcParams["axes.linewidth"] = 3
rcParams["lines.linewidth"] = 4
rcParams["lines.markersize"] = 26
rcParams["legend.loc"] = "best"
rcParams["legend.fontsize"] = 30
rcParams['axes.labelsize'] = 48
rcParams['xtick.minor.pad'] = 30.0
rcParams['xtick.labelsize'] = 48
#rcParams['ytick.minor.pad'] = -50.0


def enable_printing(figsize=(21, 9)):
    rcParams["figure.figsize"] = figsize
    rcParams["figure.dpi"] = 300.0
    rcParams["savefig.dpi"] = 'figure'
    rcParams["savefig.format"] = 'pdf'

def enable_onscreen_display():
    rcParams["figure.figsize"] = (16, 9)
    rcParams["figure.dpi"] = 100.0


def set_rcparams(**kwargs):
    for key, value in kwargs.items():
        rcParams[key] = value


def get_plot_domain(precision=None, custom_x=None):
    """
    Generates the default domain of configuration values to be plotted.
    :param precision: Number of samples per unit interval [0, 1). If None (default), uses params['sample_precision'].
    :param custom_x: (Optional) Numpy-array compatible list of x values tha tmust be included in the plot.
    :return: A NumPy-array of shape [-1, 1]
    """
    if precision is None:
        X_ = np.arange(bounds["x_intro"][0], bounds["x_intro"][1], 1 / params['sample_precision']).reshape(-1, 1)
    else:
        X_ = np.arange(bounds["x_intro"][0], bounds["x_intro"][1], 1 / precision).reshape(-1, 1)
    if custom_x is not None:
        custom_x = np.array(custom_x).reshape(-1, 1)
        logging.debug("Custom x has shape {0}".format(custom_x.shape))
        X_ = np.unique(np.vstack((X_, custom_x))).reshape(-1, 1)

    return X_


# Plot objective function, defined f(x)
def plot_objective_function(ax=None, translation=0):
    """
    Plots the underlying true objective function being used for BO.
    :param ax: matplotlib.Axes.axes object given by the user, or newly generated for a 1x1 figure if None (default).
    :param translation: int for plotting a translated objective function
    :return: None if ax was given, otherwise the new matplotlib.Axes.axes object.
    """
    return_flag = False
    if ax is None:
        fig, ax = plt.subplots(1, 1, squeeze=True)
        return_flag = True
    X_ = get_plot_domain()
    ax.plot(X_, np.add(f([X_]), translation), linestyle='--', label="Objective function")

    return ax if return_flag else None


def mark_current_incumbent(x, y, invert_y=False, ax=None, translation=0):
    """
    Convenience function to mark the current incumbent on the graph.
    :param x: Current incumbent's configuration.
    :param y: Current incumbent's observed cost.
    :param invert_y: Use the negative of the given y value, useful when switching between minimization and maximization.
    :param ax: A matplotlib.Axes.axes object on which the graphs are plotted. If None (default), a new 1x1 subplot is
    generated and the corresponding axes object is returned.
    :param translation: int for translating the coordinates along the y axis
    :return: If ax is None, the matplotlib.Axes.axes object on which plotting took place, else None.
    """

    if invert_y:
        y = -y
    ax.scatter(x, np.add(y, translation), color=colors['current_incumbent'], marker='v', label=labels['incumbent'], zorder=12)


def mark_observations(X_, Y_, mark_incumbent=True, highlight_datapoint=None, highlight_label=None, ax=None, translation=10):
    """
    Plots the given dataset as data observed thus far, including the current incumbent unless otherwise specified.
    :param X_: Configurations.
    :param Y_: Observed Costs.
    :param mark_incumbent: When True (default), distinctly marks the location of the current incumbent.
    :param highlight_datapoint: Optional array of indices of configurations in X_ which will be highlighted.
    :param highlight_label: Optional legend label for highlighted datapoints.
    :param ax: matplotlib.Axes.axes object given by the user, or newly generated for a 1x1 figure if None (default).
    :param translation: int for translating the coordinates along the y axis
    :return: None if ax was given, otherwise the new matplotlib.Axes.axes object.
    """
    return_flag = False
    if ax is None:
        fig, ax = plt.subplots(1, 1, squeeze=True)
        return_flag = True

    X_ = np.array(X_).reshape(-1, 1)
    Y_ = np.array(Y_).reshape(-1, 1)
    mask = np.ones(X_.shape[0], dtype=bool)
    logging.debug("Marking dataset with X of shape {} and Y of shape {}".format(X_.shape, Y_.shape))
    if mark_incumbent:
        incumb_idx = np.argmin(Y_)
        mark_current_incumbent(X_[incumb_idx, 0], Y_[incumb_idx, 0], ax=ax)
        mask[incumb_idx] = 0

    if highlight_datapoint is not None:
        logging.debug("Placing highlights on labels at indices: {}".format(highlight_datapoint))
        ax.scatter(
            X_[highlight_datapoint, 0],
            np.add(Y_[highlight_datapoint, 0], translation),
            color=colors['new_observation'],
            marker='X',
            label=highlight_label,
            zorder=11
        )
        mask[highlight_datapoint] = 0
    ax.scatter(X_[mask, 0], np.add(Y_[mask, 0], translation), color=colors['observations'], marker='X', label="Observations", zorder=10)

    return ax if return_flag else None


def plot_gp_samples(mu, nsamples, precision=None, custom_x=None, show_min=False, ax=None):
    """
    Plot a number of samples from a GP.
    :param mu: numpy NDArray of shape [-1, nsamples] containing samples from the GP.
    :param nsamples: Number of samples to be drawn from the GP.
    :param custom_x: (Optional) Numpy-array compatible list of x values tha tmust be included in the plot.
    :param precision: Set plotting precision per unit along x-axis. Default params['sample_precision'].
    :param show_min: If True, highlights the minima of each sample. Default False.
    :param ax: A matplotlib.Axes.axes object on which the graphs are plotted. If None (default), a new 1x1 subplot is
    generated and the corresponding axes object is returned.
    :return: If ax is None, the matplotlib.Axes.axes object on which plotting took place, else None.
    """
    return_flag = False
    if ax is None:
        fig, ax = plt.subplots(1, 1, squeeze=True)
        return_flag = True

    X_ = get_plot_domain(precision=precision, custom_x=custom_x)
    logging.debug("Generated x values for plotting of shape {0}".format(X_.shape))

    logging.debug("Plotting values for x of shape {0}".format(X_.shape))

    min_idx = np.argmin(mu, axis=0).reshape(-1, nsamples)

    rng = np.random.mtrand._rand
    if seed is not None:
        rng = np.random.RandomState(seed)

    xmin = []
    mumin = []
    for i in range(nsamples):
        ax.plot(X_, mu[:, i], color=rng.rand(3), label="Sample {}".format(i+1), alpha=0.6,)
        xmin.append(X_[min_idx[0, i], 0])
        mumin.append(mu[min_idx[0, i], i])
    if show_min:
        ax.scatter(
            xmin,
            mumin,
            color=colors['highlighted_observations'],
            marker='X',
            label='Sample Minima',
            zorder=11
        )

    return ax if return_flag else None



def plot_gp(model, confidence_intervals=None, type='both', custom_x=None, precision=None, ax=None, translation=0, annotate=False):
    """
    Plot a GP's mean and, if required, its confidence intervals.
    :param model: GP
    :param confidence_intervals: If None (default) no confidence envelope is plotted. If a list of positive values
    [k1, k2, ...]is given, the confidence intervals k1*sigma, k2*sigma, ... are plotted.
    :param type: 'upper'|'lower'|'both' (default) - Type of confidence bound to plot.
    :param custom_x: (Optional) Numpy-array compatible list of x values that must be included in the plot.
    :param precision: Set plotting precision per unit along x-axis. Default params['sample_precision'].
    :param ax: A matplotlib.Axes.axes object on which the graphs are plotted. If None (default), a new 1x1 subplot is
    generated and the corresponding axes object is returned.
    :param translation: int for translating the coordinates along the y axis
    :param annotate: False, If True annotations are added for the Posterior Mean and Uncertainty
    :return: If ax is None, the matplotlib.Axes.axes object on which plotting took place, else None.
    """
    return_flag = False
    if ax is None:
        fig, ax = plt.subplots(1, 1, squeeze=True)
        return_flag = True

    X_ = get_plot_domain(precision=precision, custom_x=custom_x)
    logging.debug("Generated x values for plotting of shape {0}".format(X_.shape))


    def draw_confidence_envelopes(mu, sigma, confidence_intervals):
        confidence_intervals = np.array(confidence_intervals)
        confidence_intervals.sort()

        # Dynamically generate opacities for each confidence envelope
        alphas = np.linspace(
            start=colors['envelope_max_opacity'],
            stop=colors['envelope_min_opacity'],
            num=confidence_intervals.shape[0],
            endpoint=False
        )

        get_envelope = {
            'upper': lambda mu, k, sigma: (mu, mu + k * sigma),
            'lower': lambda mu, k, sigma: (mu - k * sigma, mu),
            'both': lambda mu, k, sigma: (mu - k * sigma, mu + k * sigma),
        }

        for k, alpha in zip(confidence_intervals, alphas):
            lower, upper = get_envelope[type](mu, k, sigma)
            ax.fill_between(
                X_[:, 0], lower + translation, upper + translation,
                facecolor=colors['gp_variance'], alpha=alpha,
                label="{0:.2f}-Sigma Confidence Envelope".format(k)
            )


    if annotate:
        annotate_x = [6, 8.5]
        X_predict = np.vstack((X_, [[annotate_x[0]]])).reshape(-1, 1)
        X_predict = np.vstack((X_predict, [[annotate_x[1]]])).reshape(-1, 1)
        mu, sigma = model.predict(X_predict, return_std=True)
    else:
        mu, sigma = model.predict(X_, return_std=True)
    logging.debug("Plotting GP with these values:\nSamples:\t\t{0}\nMeans:\t\t{1}\nSTDs:\t\t{2}".format(
        X_, mu, sigma
    ))

    # Plot the mean
    if annotate:
        ax.plot(X_, np.add(mu[:-2], translation), color=colors['gp_mean'], label=labels['gp_mean'])
    else:
        ax.plot(X_, np.add(mu, translation), color=colors['gp_mean'], label=labels['gp_mean'])

    # If needed, plot the confidence envelope(s)
    if confidence_intervals is not None:
        if annotate:
            draw_confidence_envelopes(mu[:-2], sigma[:-2], confidence_intervals)
        else:
            draw_confidence_envelopes(mu, sigma, confidence_intervals)
    if annotate:
        ax.annotate("Posterior mean", xy=(annotate_x[0], mu[-2] + 10), xytext=(annotate_x[0] - 1.15, mu[-2]+ 5),
                         arrowprops={'arrowstyle': 'fancy'}, zorder=19, fontsize='x-large')
        ax.annotate("Posterior uncertainty", xy=(annotate_x[1], mu[-1]- sigma[-1] + 10), xytext=(annotate_x[1] - 0.7, mu[-1] - sigma[-1] + 6),
                         arrowprops={'arrowstyle': 'fancy'}, zorder=10, fontsize='x-large')
    return ax if return_flag else None


# Plot acquisition function
def plot_acquisition_function(acquisition, eta, model, add=None, invert=False, ax=None, annotate=False, scaling=1):
    """
    Generate a plot to visualize the given acquisition function for the model.
    :param acquisition: Acquisition function handle, from bo_configurations.acquisition_functions.
    :param eta: Best observed value thus far.
    :param model: GP to be used as a model.
    :param add: Additional parameters passed to the acquisition function.
    :param invert: When True (default), it is assumed that the acquisition function needs to be inverted for plotting.
    :param ax: A matplotlib.Axes.axes object on which the graphs are plotted. If None (default), a new 1x1 subplot is
    generated and the corresponding axes object is returned.
    :param annotate: False, If True annotations are added for the Acquisition Function and Acquisition Function Max
    :param scaling: int for plotting a scaled acquisition function
    :return: If ax is None, the matplotlib.Axes.axes object on which plotting took place, else None.
    """
    return_flag = False
    if ax is None:
        fig, ax = plt.subplots(1, 1, squeeze=True)

        ax.set_xlim(bounds["x_intro"])
        ax.set_ylim(bounds["y_intro"])
        ax.grid()
        ax.set_xlabel(labels['xlabel'])
        ax.set_ylabel(labels['acq_ylabel'])
        ax.set_title(r"Visualization of {}".format(labels[acquisition]), loc='left')

        return_flag = True

    X_ = get_plot_domain().reshape(-1)

    if annotate:
        np.hstack((X_, [6])).reshape(-1, 1)


    acquisition_fun = acquisition_functions[acquisition](X_, model=model, eta=eta, add=add)
    if invert:
        acquisition_fun = -acquisition_fun
    zipped = list(zip(X_, acquisition_fun))
    zipped.sort(key = lambda t: t[0])
    X_, acquisition_fun = list(zip(*zipped))

    ax.plot(X_, np.clip(acquisition_fun, a_min=-2, a_max=5)*scaling, color=colors['acq_func_intro'], label=labels[acquisition])
    ax.fill_between(X_, np.clip(acquisition_fun, a_min=-2, a_max=5)*scaling, bounds["y_intro"][0], facecolor=colors['acq_func_intro_fill'])
    acq_vals = np.clip(acquisition_fun, a_min=-2, a_max=5)*scaling
    best = np.argmax(acq_vals)

    if annotate:
        ax.annotate("Acquisition function", xy=(6, acq_vals[-1]),
                     xytext=(6, acq_vals[-1] + 2),
                     arrowprops={'arrowstyle': 'fancy'}, zorder=10, fontsize='x-large')
        ax.annotate("Acquisition max", xy=(X_[best], acq_vals[best]),
                    xytext=(X_[best] -2.45, acq_vals[best] + 1),
                    arrowprops={'arrowstyle': 'fancy'}, zorder=10, fontsize='x-large')

    return ax if return_flag else None



def highlight_configuration(x, label=None, lloc='bottom', ax=None, disable_ticks=False, **kwargs):
    """
    Draw a vertical line at the given configuration to highlight it.
    :param x: Configuration.
    :param label: If None (default), the x-value up to decimal places is placed as a minor tick, otherwise the given
    label is used.
    :param lloc: Can be either 'top' or 'bottom' (default) to indicate the position of the label on the graph.
    :param ax: A matplotlib.Axes.axes object on which the graphs are plotted. If None (default), a new 1x1 subplot is
    generated and the corresponding axes object is returned.
    :param disable_ticks: Only draw the horizontal line, don't bother with the ticks.
    :return: If ax is None, the matplotlib.Axes.axes object on which plotting took place, else None.
    """
    return_flag = False
    if ax is None:
        fig, ax = plt.subplots(1, 1, squeeze=True)
        return_flag = True

    # Assume we will recieve x as a view on a numpy array
    x = x.reshape(-1)[0]
    logging.info("Highlighting configuration at {} with label {}".format(x, label))

    ax.vlines(
        x, ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1],
        colors=colors['minor_tick_highlight'], linestyles='dashed',
    )

    if disable_ticks:
        return ax if return_flag else None

    xlabel = "{0:.2f}".format(x) if label is None else label

    if lloc == 'top':
        ax.tick_params(
            which='minor',
            bottom=False, labelbottom=False,
            top=True, labeltop=True
        )
    else:
        ax.tick_params(
            which='minor',
            bottom=True, labelbottom=True,
            top=False, labeltop=False
        )

    label_props = {'color': colors['minor_tick_highlight'], **kwargs}
    ax.set_xticks([x], minor=True)
    ax.set_xticklabels([xlabel], label_props, minor=True)

    return ax if return_flag else None


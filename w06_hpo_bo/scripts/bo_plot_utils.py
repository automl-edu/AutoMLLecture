from matplotlib import pyplot as plt
import numpy as np
import logging

from bo_configurations import *
from matplotlib import rcParams
from matplotlib.patches import Rectangle

from scipy.stats import norm

rcParams["font.size"] = 32
rcParams["axes.linewidth"] = 3
rcParams["lines.linewidth"] = 4
rcParams["lines.markersize"] = 26
rcParams["legend.loc"] = "lower right"
rcParams["legend.fontsize"] = 26
rcParams['axes.labelsize'] = 36
rcParams['xtick.minor.pad'] = 30.0
#rcParams['ytick.minor.pad'] = -50.0



def enable_printing():
    rcParams["figure.figsize"] = (21, 9)
    rcParams["figure.dpi"] = 300.0
    rcParams["savefig.dpi"] = 'figure'
    rcParams["savefig.format"] = 'pdf'

def enable_onscreen_display():
    rcParams["figure.figsize"] = (16, 9)
    rcParams["figure.dpi"] = 100.0


def set_rcparams(**kwargs):
    for key, value in kwargs.items():
        rcParams[key] = value


def annotate_y_edge(label, xy, ax, align='right'):
    """
    Place an annotation beneath a horizontal bar, between a given point and either of the left or right edges.
    :param label: Text to annotate with.
    :param xy: Given xy-coordinates.
    :param ax: matplotlib.Axes.axes object given by the user
    :param align: 'left' or 'right' (default) edge to use.
    :return: None.
    """

    if align == 'left':
        x = xy[0] - (xy[0] - ax.get_xlim()[0]) / 2
    else:
        x = xy[0] + (ax.get_xlim()[1] - xy[0]) / 2

    # textxy = ax.transData.transform([x, xy[1]])
    # textxy = ax.transData.inverted().transform((textxy[0], textxy[1] - 2 * rcParams["font.size"]))
    textxy = (x, xy[1] - (ax.get_ylim()[1] - ax.get_ylim()[0]) / 10)
    # logging.info("Placing text at {}".format(textxy))

    ax.annotate(s=label, xy=textxy, color=colors['minor_tick_highlight'], horizontalalignment='center', zorder=10)


def annotate_x_edge(label, xy, ax, align='bottom', offset_param=1.5):
    """
    Place an annotation next to a vertical bar, between a given point and either of the bottom or top edges.
    :param label: Text to annotate with.
    :param xy: Given xy-coordinates.
    :param ax: matplotlib.Axes.axes object given by the user
    :param align: 'top' or 'bottom' (default) edge to use.
    :param offset_param: Controls offset of label from xy.
    :return: None.
    """

    if align == 'bottom':
        y = xy[1] - offset_param * (xy[1] - ax.get_ylim()[0]) / 2
    else:
        y = xy[1] + offset_param * (ax.get_ylim()[1] - xy[1]) / 2

    # textxy = ax.transData.transform([x, xy[1]])
    # textxy = ax.transData.inverted().transform((textxy[0], textxy[1] - 2 * rcParams["font.size"]))
    textxy = (xy[0] - 0.1, y)
    # logging.info("Placing text at {}".format(textxy))

    ax.annotate(s=label, xy=textxy, color=colors['minor_tick_highlight'], horizontalalignment='right', zorder=10)


def get_plot_domain(precision=None, custom_x=None):
    """
    Generates the default domain of configuration values to be plotted.
    :param precision: Number of samples per unit interval [0, 1). If None (default), uses params['sample_precision'].
    :param custom_x: (Optional) Numpy-array compatible list of x values tha tmust be included in the plot.
    :return: A NumPy-array of shape [-1, 1]
    """
    if precision is None:
        X_ = np.arange(bounds["x"][0], bounds["x"][1], 1 / params['sample_precision']).reshape(-1, 1)
    else:
        X_ = np.arange(bounds["x"][0], bounds["x"][1], 1 / precision).reshape(-1, 1)
    if custom_x is not None:
        custom_x = np.array(custom_x).reshape(-1, 1)
        logging.debug("Custom x has shape {0}".format(custom_x.shape))
        X_ = np.unique(np.vstack((X_, custom_x))).reshape(-1, 1)

    return X_


# Plot objective function, defined f(x)
def plot_objective_function(ax=None):
    """
    Plots the underlying true objective function being used for BO.
    :param ax: matplotlib.Axes.axes object given by the user, or newly generated for a 1x1 figure if None (default).
    :return: None if ax was given, otherwise the new matplotlib.Axes.axes object.
    """
    return_flag = False
    if ax is None:
        fig, ax = plt.subplots(1, 1, squeeze=True)
        return_flag = True
    X_ = get_plot_domain()
    ax.plot(X_, f([X_]), linestyle='--', label="Objective function")

    return ax if return_flag else None


def mark_current_incumbent(x, y, invert_y=False, ax=None):
    """
    Convenience function to mark the current incumbent on the graph.
    :param x: Current incumbent's configuration.
    :param y: Current incumbent's observed cost.
    :param invert_y: Use the negative of the given y value, useful when switching between minimization and maximization.
    :param ax: A matplotlib.Axes.axes object on which the graphs are plotted. If None (default), a new 1x1 subplot is
    generated and the corresponding axes object is returned.
    :return: If ax is None, the matplotlib.Axes.axes object on which plotting took place, else None.
    """

    if invert_y:
        y = -y
    ax.scatter(x, y, color=colors['current_incumbent'], marker='v', label=labels['incumbent'], zorder=12)


def mark_observations(X_, Y_, mark_incumbent=True, highlight_datapoint=None, highlight_label=None, ax=None):
    """
    Plots the given dataset as data observed thus far, including the current incumbent unless otherwise specified.
    :param X_: Configurations.
    :param Y_: Observed Costs.
    :param mark_incumbent: When True (default), distinctly marks the location of the current incumbent.
    :param highlight_datapoint: Optional array of indices of configurations in X_ which will be highlighted.
    :param highlight_label: Optional legend label for highlighted datapoints.
    :param ax: matplotlib.Axes.axes object given by the user, or newly generated for a 1x1 figure if None (default).
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
            Y_[highlight_datapoint, 0],
            color=colors['highlighted_observations'],
            marker='X',
            label=highlight_label,
            zorder=11
        )
        mask[highlight_datapoint] = 0
    ax.scatter(X_[mask, 0], Y_[mask, 0], color=colors['observations'], marker='X', label="Observations", zorder=10)

    return ax if return_flag else None


def plot_gp_samples(mu, nsamples, precision=None, custom_x=None, show_min=False, ax=None, seed=None):
    """
    Plot a number of samples from a GP.
    :param mu: numpy NDArray of shape [-1, nsamples] containing samples from the GP.
    :param nsamples: Number of samples to be drawn from the GP.
    :param custom_x: (Optional) Numpy-array compatible list of x values tha tmust be included in the plot.
    :param precision: Set plotting precision per unit along x-axis. Default params['sample_precision'].
    :param show_min: If True, highlights the minima of each sample. Default False.
    :param ax: A matplotlib.Axes.axes object on which the graphs are plotted. If None (default), a new 1x1 subplot is
    generated and the corresponding axes object is returned.
    :param seed: If None, uses global np.random, else uses the given seed to generate np.random.RandomState
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



def plot_gp(model, confidence_intervals=None, type='both', custom_x=None, precision=None, ax=None):
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
                X_[:, 0], lower, upper,
                facecolor=colors['gp_variance'], alpha=alpha,
                label="{0:.2f}-Sigma Confidence Envelope".format(k)
            )


    mu, sigma = model.predict(X_, return_std=True)
    logging.debug("Plotting GP with these values:\nSamples:\t\t{0}\nMeans:\t\t{1}\nSTDs:\t\t{2}".format(
        X_, mu, sigma
    ))

    # Plot the mean
    ax.plot(X_, mu, color=colors['gp_mean'], label=labels['gp_mean'])

    # If needed, plot the confidence envelope(s)
    if confidence_intervals is not None:
        draw_confidence_envelopes(mu, sigma, confidence_intervals)

    return ax if return_flag else None


# Plot acquisition function
def plot_acquisition_function(acquisition, eta, model, add=None, ax=None):
    """
    Generate a plot to visualize the given acquisition function for the model. It is assumed that the acquisition
    function will always return a value for the minimizer, and will hence be negated before plotting.
    :param acquisition: Acquisition function handle, from bo_configurations.acquisition_functions.
    :param eta: Best observed value thus far.
    :param model: GP to be used as a model.
    :param add: Additional parameters passed to the acquisition function.
    :param ax: A matplotlib.Axes.axes object on which the graphs are plotted. If None (default), a new 1x1 subplot is
    generated and the corresponding axes object is returned.
    :return: If ax is None, the matplotlib.Axes.axes object on which plotting took place, else None.
    """
    return_flag = False
    if ax is None:
        fig, ax = plt.subplots(1, 1, squeeze=True)

        ax.set_xlim(bounds["x"])
        ax.set_ylim(bounds["acq_y"])
        ax.grid()
        ax.set_xlabel(labels['xlabel'])
        ax.set_ylabel(labels['acq_ylabel'])
        ax.set_title(r"Visualization of {}".format(labels[acquisition]), loc='left')

        return_flag = True

    X_ = get_plot_domain().reshape(-1)
    acquisition_fun = acquisition_functions[acquisition](X_, model=model, eta=eta, add=add)
    acquisition_fun = -acquisition_fun
    zipped = list(zip(X_, acquisition_fun))
    zipped.sort(key = lambda t: t[0])
    X_, acquisition_fun = list(zip(*zipped))

    ax.plot(X_, acquisition_fun, color=colors['acq_fun'], label=labels[acquisition])
    ax.fill_between(X_, acquisition_fun, ax.get_ylim()[0], facecolor=colors['acq_func_fill'])

    return ax if return_flag else None

    # now = datetime.now()
    # dt_string = now.strftime("%d_%m_%Y__%H_%M_%S.%f")
    # plt.savefig("../plots/bo_loop/" + dt_string + ".png")
    # plt.clf()


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

def highlight_output(y, label=None, lloc='left', ax=None, disable_ticks=False, **kwargs):
    """
    Draw a horizontal line at the given y-value to highlight it.
    :param y: y-value to be highlighted.
    :param label: If None (default), the y-value up to decimal places is placed as a minor tick, otherwise the given
    label is used.
    :param lloc: Can be either 'left' (default) or 'right' to indicate the position of the label on the graph.
    :param ax: A matplotlib.Axes.axes object on which the graphs are plotted. If None (default), a new 1x1 subplot is
    generated and the corresponding axes object is returned.
    :param disable_ticks: Only draw the horizontal line, don't bother with the ticks.
    :return: If ax is None, the matplotlib.Axes.axes object on which plotting took place, else None.
    """
    return_flag = False
    if ax is None:
        fig, ax = plt.subplots(1, 1, squeeze=True)
        return_flag = True

    # Assume we will recieve y as a view on a numpy array
    y = y.reshape(-1)[0]

    ax.hlines(
        y,
        xmin=ax.get_xlim()[0], xmax=ax.get_xlim()[1],
        colors=colors['minor_tick_highlight'], linestyles='dashed'
    )

    if disable_ticks:
        return ax if return_flag else None

    if lloc == 'right':
        ax.tick_params(
            which='minor',
            left=False, labelleft=False,
            right=True, labelright=True
        )
    else:
        ax.tick_params(
            which='minor',
            left=True, labelleft=True,
            right=False, labelright=False
        )

    ylabel = "{0:.2f}".format(y) if label is None else label
    label_props = {'color': colors['minor_tick_highlight'], **kwargs}
    ax.set_yticks([y], minor=True)
    ax.set_yticklabels([ylabel], label_props, minor=True)

    return ax if return_flag else None

def darken_graph(y, ax):
    """
    Darken the graph above a certain y-value.
    :param y: Boundary between light and dark.
    :param ax: The matplotlib.axes.Axes object to work on.
    :return: None
    """

    recto = (ax.get_xlim()[0], y)
    rectwidth = ax.get_xlim()[1] - ax.get_xlim()[0]
    rectheight = ax.get_ylim()[1] - y
    rect = Rectangle(
        recto, rectwidth, rectheight,
        fill=True, alpha=0.75, facecolor='white', zorder=8, linewidth=rcParams['lines.linewidth'], edgecolor='grey'
    )
    ax.add_patch(rect)
    return


def draw_vertical_normal(gp, incumbenty, ax, xtest=0.0, step=0.01,
                         xlim=2.0, xscale=1.0, yscale=1.0):

    # Generate a normal pdf centered at xtest
    ytest_mean, ytest_cov = gp.predict([[xtest]], return_cov=True)
    mu = ytest_mean[0]
    sigma = ytest_cov[0, 0]
    logging.info("Shapes:\nytext_mean:{}\nytest_conv:{}\nmu:{}\sigma:{}".format(
        ytest_mean.shape, ytest_cov.shape, mu.shape, sigma.shape
    ))
    logging.info("Generating N({},{})".format(mu, sigma))
    # print("ytest mean:{}, cov:{}".format(ytest_mean, ytest_cov))

    # Generate a Normal distribution centered around it's mean.
    norm_x = np.arange(mu - xlim, mu + xlim + step, step)
    norm_y = norm.pdf(norm_x, mu, sigma) * yscale
    logging.info("Min of normal_y is: {}\nMean of normal_y is:{}".format(np.min(norm_y), np.mean(norm_y)))

    # Shift the normal to x=0 and stretch it along x
    norm_x = (norm_x - mu) * xscale

    # Rotate by -pi/2 to obtain a vertical curve and then shift the center to the sample's location on the GP Mean
    vcurve_x = norm_y + xtest
    vcurve_y = -norm_x + mu

    # Test the curve by centering it without rotation
    # vcurve_x = norm_x + xtest
    # vcurve_y = norm_y + mu

    ax.plot(xtest, mu, color='red', marker='o', markersize=20, zorder=14)
    ax.vlines(xtest, ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1], colors='black', linestyles='dashed', zorder=9)
    ax.plot(vcurve_x, vcurve_y, color='black', zorder=9)
    fill_args = np.where(vcurve_y < incumbenty)
    ax.fill_betweenx(vcurve_y[fill_args], xtest, vcurve_x[fill_args], alpha=1.0, facecolor='darkgreen', zorder=14)

    # ann_x = xtest
    # ann_y = mu
    #
    # arrow_x = ann_x - 2.0
    # arrow_y = ann_y - 1.5
    #
    # ax.annotate(s=r'$({0}, \mu({0}))$'.format(label), xy=(ann_x, ann_y), xytext=(arrow_x, arrow_y),
    #             arrowprops={'arrowstyle': 'fancy'},
    #              weight='heavy', zorder=15)

    return (vcurve_x, vcurve_y, mu)

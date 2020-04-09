import warnings
warnings.filterwarnings('ignore')
import argparse
import logging

import numpy as np
from sklearn.neighbors import KernelDensity as kd
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import Matern

from matplotlib import pyplot as plt

import bo_plot_utils as boplot
from bo_configurations import *

SEED = None
TOGGLE_PRINT = False
INIT_X_PRESENTATION = [2.5, 3.5, 5.5, 7, 9]

labels["xlabel"] = "$\lambda'$"
labels["ylabel"] = "$c(\lambda')$"
bounds["acq_y"] = (0.0, 10.0)
bounds["x"] = (2, 13)
bounds["gp_y"] = (-5, 5)

def initialize_dataset(initial_design, init=None):
    """
    Initialize some data to start fitting the GP on.
    :param initial_design: Method for initializing the GP, choice between 'uniform', 'random', and 'presentation'
    :param init: Number of datapoints to initialize with, if relevant
    :return:
    """

    x = None

    # sample initial query points
    if initial_design == 'uniform':
        x = np.linspace(bounds['x'][0], bounds['x'][1], init).reshape(-1, 1).tolist()
    elif initial_design == 'random':
        x = np.random.uniform(bounds['x'][0], bounds['x'][1], init).reshape(-1, 1).tolist()
    elif initial_design == 'presentation':
        x = np.array(INIT_X_PRESENTATION).reshape(-1, 1).tolist()

    # get corresponding response values
    y = list(map(f, x))

    return x, y


def get_mu_star(model):
    """
    Given a model, return the (x, y) coords of mu-star.
    :param model: The underlying GP.
    :return:
    """

    X_ = boplot.get_plot_domain()
    mu = model.predict(X_).reshape(-1, 1)

    coords = np.hstack((X_, mu))
    idx = np.argmin(coords, axis=0)[1]
    return coords[idx, :]


def visualize_es(initial_design, init=None):
    """
    Visualize one-step of ES.
    :param initial_design: Method for initializing the GP, choice between 'uniform', 'random', and 'presentation'
    :param init: Number of datapoints to initialize GP with.
    :return: None
    """

    # 1. Show GP fit on initial dataset, 0 samples, histogram
    # 2. Show GP fit on initial dataset, 1 sample, histogram
    # 3. Show GP fit on initial dataset, 3 samples, histogram
    # 4. Show GP fit on initial dataset, 50 samples, histogram
    # 5. Show PDF derived from the histogram at 50 samples
    # 6. Mark maximum of the PDF as next configuration to be evaluated


    # a. Plot GP
    # b. Sample GP, mark minima, update histogram of lambda*
    # c. Repeat 2 for each sample.
    # d. Show results after multiple iterations


    boplot.set_rcparams(**{'figure.figsize': (22, 11)})

    # Initial setup
    # -------------------------------------------

    logging.debug("Visualizing ES with initial design {} and init {}".format(initial_design, init))
    # Initialize dummy dataset
    x, y = initialize_dataset(initial_design=initial_design, init=init)
    logging.debug("Initialized dataset with:\nsamples {0}\nObservations {1}".format(x, y))

    # Fit GP to the currently available dataset
    gp = GPR(kernel=Matern())
    logging.debug("Fitting GP to\nx: {}\ny:{}".format(x, y))
    gp.fit(x, y)  # fit the model

    histogram_precision = 20
    X_ = boplot.get_plot_domain(precision=histogram_precision)
    nbins = X_.shape[0]
    logging.info("Creating histograms with {} bins".format(nbins))
    bin_range = (bounds['x'][0], bounds['x'][1] + 1 / histogram_precision)

    # -------------------------------------------

    def draw_samples(nsamples, ax1, ax2, show_min=False, return_pdf=False):
        if not nsamples:
            return
        seed2 = 1256
        seed3 = 65

        mu = gp.sample_y(X=X_, n_samples=nsamples, random_state=seed3)
        boplot.plot_gp_samples(
            mu=mu,
            nsamples=nsamples,
            precision=histogram_precision,
            custom_x=X_,
            show_min=show_min,
            ax=ax1,
            seed=seed2
        )
        data_h = X_[np.argmin(mu, axis=0), 0]
        logging.info("Shape of data_h is {}".format(data_h.shape))
        logging.debug("data_h is: {}".format(data_h))

        bins = ax2.hist(
            data_h, bins=nbins,
            range=bin_range, density=return_pdf,
            color='lightgreen', edgecolor='black', alpha=0.0 if return_pdf else 1.0
        )

        return bins

    # 1. Show GP fit on initial dataset, 0 samples, histogram
    # -------------------------------------------

    ax2_title = r'$p_{min}=P(\lambda=\lambda^*)$'

    bounds['acq_y'] = (0.0, 1.0)

    fig, (ax1, ax2) = plt.subplots(2, 1, squeeze=True)
    ax1.set_xlim(bounds['x'])
    ax1.set_ylim(bounds['gp_y'])
    ax2.set_xlim(bounds['x'])
    ax2.set_ylim(bounds['acq_y'])
    ax1.grid()
    ax2.grid()

    boplot.plot_objective_function(ax=ax1)
    boplot.plot_gp(model=gp, confidence_intervals=[3.0], ax=ax1, custom_x=x)
    boplot.mark_observations(X_=x, Y_=y, mark_incumbent=False, ax=ax1)

    nsamples = 0
    draw_samples(nsamples=nsamples, ax1=ax1, ax2=ax2, show_min=True)

    # Plot uniform prior for p_min
    xplot = boplot.get_plot_domain()
    ylims = ax2.get_ylim()
    xlims = ax2.get_xlim()
    yupper = [(ylims[1] - ylims[0]) / (xlims[1] - xlims[0])] * xplot.shape[0]
    ax2.plot(xplot[:, 0], yupper, color='green', linewidth=2.0)
    ax2.fill_between(xplot[:, 0], ylims[0], yupper, color='lightgreen')

    ax1.legend().set_zorder(20)
    ax1.set_xlabel(labels['xlabel'])
    ax1.set_ylabel(labels['gp_ylabel'])
    ax1.set_title(r"Visualization of $\mathcal{G}^t$", loc='left')

    ax2.set_xlabel(labels['xlabel'])
    ax2.set_ylabel(r'$p_{min}$')
    ax2.set_title(ax2_title, loc='left')

    plt.tight_layout()
    if TOGGLE_PRINT:
        plt.savefig('es_1')
    else:
        plt.show()
    # -------------------------------------------

    # 2. Show GP fit on initial dataset, 1 sample, histogram
    # -------------------------------------------

    bounds['acq_y'] = (0.0, 5.0)
    ax2_title = r'Frequency of $\lambda=\hat{\lambda}^*$'

    fig, (ax1, ax2) = plt.subplots(2, 1, squeeze=True)
    ax1.set_xlim(bounds['x'])
    ax1.set_ylim(bounds['gp_y'])
    ax2.set_xlim(bounds['x'])
    ax2.set_ylim(bounds['acq_y'])
    ax1.grid()
    ax2.grid()

    boplot.plot_objective_function(ax=ax1)
    boplot.mark_observations(X_=x, Y_=y, mark_incumbent=False, ax=ax1)

    nsamples = 1
    draw_samples(nsamples=nsamples, ax1=ax1, ax2=ax2, show_min=True)

    ax1.legend().set_zorder(20)
    ax1.set_xlabel(labels['xlabel'])
    ax1.set_ylabel(labels['gp_ylabel'])
    ax1.set_title(r"One sample from $\mathcal{G}^t$", loc='left')

    ax2.set_xlabel(labels['xlabel'])

    # ax2.set_ylabel(r'$p_{min}$')
    ax2.set_ylabel(r'Frequency')
    ax2.set_title(ax2_title, loc='left')

    plt.tight_layout()
    if TOGGLE_PRINT:
        plt.savefig('es_2')
    else:
        plt.show()

    # 3. Show GP fit on initial dataset, 10 samples, histogram
    # -------------------------------------------

    bounds['acq_y'] = (0.0, 10.0)
    ax2_title = r'Frequency of $\lambda=\hat{\lambda}^*$'

    fig, (ax1, ax2) = plt.subplots(2, 1, squeeze=True)
    ax1.set_xlim(bounds['x'])
    ax1.set_ylim(bounds['gp_y'])
    ax2.set_xlim(bounds['x'])
    ax2.set_ylim(bounds['acq_y'])
    ax1.grid()
    ax2.grid()

    boplot.plot_objective_function(ax=ax1)
    boplot.mark_observations(X_=x, Y_=y, mark_incumbent=False, ax=ax1)

    nsamples = 10
    draw_samples(nsamples=nsamples, ax1=ax1, ax2=ax2)

    ax1.set_xlabel(labels['xlabel'])
    ax1.set_ylabel(labels['gp_ylabel'])
    ax1.set_title(r"Ten samples from $\mathcal{G}^t$", loc='left')

    ax2.set_xlabel(labels['xlabel'])

    # ax2.set_ylabel(r'$p_{min}$')
    ax2.set_ylabel(r'Frequency')
    ax2.set_title(ax2_title, loc='left')

    plt.tight_layout()
    if TOGGLE_PRINT:
        plt.savefig('es_3')
    else:
        plt.show()

    # -------------------------------------------

    # 4. Show GP fit on initial dataset, 200 samples, histogram
    # -------------------------------------------

    bounds["acq_y"] = (0.0, 20.0)
    ax2_title = r'Frequency of $\lambda=\hat{\lambda}^*$'

    fig, (ax1, ax2) = plt.subplots(2, 1, squeeze=True)
    ax1.set_xlim(bounds['x'])
    ax1.set_ylim(bounds['gp_y'])
    ax2.set_xlim(bounds['x'])
    ax2.set_ylim(bounds['acq_y'])
    ax1.grid()
    ax2.grid()

    boplot.plot_objective_function(ax=ax1)
    boplot.mark_observations(X_=x, Y_=y, mark_incumbent=False, ax=ax1)

    nsamples = 200
    draw_samples(nsamples=nsamples, ax1=ax1, ax2=ax2)

    ax1.set_xlabel(labels['xlabel'])
    ax1.set_ylabel(labels['gp_ylabel'])
    ax1.set_title(r"200 samples from $\mathcal{G}^t$", loc='left')

    ax2.set_xlabel(labels['xlabel'])

    # ax2.set_ylabel(r'$p_{min}$')
    ax2.set_ylabel(r'Frequency')
    ax2.set_title(ax2_title, loc='left')

    plt.tight_layout()
    if TOGGLE_PRINT:
        plt.savefig('es_4')
    else:
        plt.show()
    # -------------------------------------------

    # 5. Show PDF derived from the histogram at 200 samples
    # -------------------------------------------

    ax2_title = "$\hat{P}(\lambda=\lambda^*)$"
    bounds["acq_y"] = (0.0, 1.0)

    fig, (ax1, ax2) = plt.subplots(2, 1, squeeze=True)
    ax1.set_xlim(bounds['x'])
    ax1.set_ylim(bounds['gp_y'])
    ax2.set_xlim(bounds['x'])
    ax2.set_ylim(bounds["acq_y"])
    ax1.grid()
    ax2.grid()

    boplot.plot_objective_function(ax=ax1)
    boplot.plot_gp(model=gp, confidence_intervals=[3.0], ax=ax1, custom_x=x)
    boplot.mark_observations(X_=x, Y_=y, mark_incumbent=False, ax=ax1)

    nsamples = 200
    seed3 = 65

    mu = gp.sample_y(X=X_, n_samples=nsamples, random_state=seed3)
    data_h = X_[np.argmin(mu, axis=0), 0]

    kde = kd(kernel='gaussian', bandwidth=0.75).fit(data_h.reshape(-1, 1))
    xplot = boplot.get_plot_domain()
    ys = np.exp(kde.score_samples(xplot))

    ax2.plot(xplot, ys, color='green', lw=2.)
    ax2.fill_between(xplot[:, 0], ax2.get_ylim()[0], ys, color='lightgreen')

    ax1.set_xlabel(labels['xlabel'])
    ax1.set_ylabel(labels['gp_ylabel'])
    ax1.set_title(r"Visualization of $\mathcal{G}^t$", loc='left')

    ax2.set_xlabel(labels['xlabel'])
    ax2.set_ylabel(r'$p_{min}$')
    ax2.set_title(ax2_title, loc='left')

    plt.tight_layout()
    if TOGGLE_PRINT:
        plt.savefig('es_5')
    else:
        plt.show()

    # -------------------------------------------

    # 6. Mark maximum of the PDF as next configuration to be evaluated
    # -------------------------------------------

    ax2_title = "$\hat{P}(\lambda=\lambda^*)$"
    bounds["acq_y"] = (0.0, 1.0)

    fig, (ax1, ax2) = plt.subplots(2, 1, squeeze=True)
    ax1.set_xlim(bounds['x'])
    ax1.set_ylim(bounds['gp_y'])
    ax2.set_xlim(bounds['x'])
    ax2.set_ylim(bounds["acq_y"])
    ax1.grid()
    ax2.grid()

    boplot.plot_objective_function(ax=ax1)
    boplot.plot_gp(model=gp, confidence_intervals=[3.0], ax=ax1, custom_x=x)
    boplot.mark_observations(X_=x, Y_=y, mark_incumbent=False, ax=ax1)

    nsamples = 200
    seed3 = 65

    mu = gp.sample_y(X=X_, n_samples=nsamples, random_state=seed3)
    data_h = X_[np.argmin(mu, axis=0), 0]

    kde = kd(kernel='gaussian', bandwidth=0.75).fit(data_h.reshape(-1, 1))
    xplot = boplot.get_plot_domain()
    ys = np.exp(kde.score_samples(xplot))

    idx_umax = np.argmax(ys)
    boplot.highlight_configuration(x=xplot[idx_umax], label='', ax=ax1, disable_ticks=True)
    boplot.annotate_x_edge(label=r'$\lambda^{(t)}$', xy=(xplot[idx_umax], ax1.get_ylim()[0]),
                           ax=ax1, align='top', offset_param= 1.5)
    boplot.highlight_configuration(x=xplot[idx_umax], label='', ax=ax2, disable_ticks=True)
    boplot.annotate_x_edge(label=r'$\lambda^{(t)}$', xy=(xplot[idx_umax], ys[idx_umax]),
                           ax=ax2, align='top', offset_param=1.0)

    ax2.plot(xplot, ys, color='green', lw=2.)
    ax2.fill_between(xplot[:, 0], ax2.get_ylim()[0], ys, color='lightgreen')

    ax1.set_xlabel(labels['xlabel'])
    ax1.set_ylabel(labels['gp_ylabel'])
    ax1.set_title(r"Visualization of $\mathcal{G}^t$", loc='left')

    ax2.set_xlabel(labels['xlabel'])
    ax2.set_ylabel(r'$p_{min}$')
    ax2.set_title(ax2_title, loc='left')

    plt.tight_layout()
    if TOGGLE_PRINT:
        plt.savefig('es_6')
    else:
        plt.show()

    # -------------------------------------------


def main(init_size, initial_design):
        visualize_es(
            init=init_size,
            initial_design=initial_design,
        )



if __name__ == '__main__':
    cmdline_parser = argparse.ArgumentParser('AutoMLLecture')

    cmdline_parser.add_argument('-f', '--init_db_size',
                                default=4,
                                help='Size of the initial database',
                                type=int)
    cmdline_parser.add_argument('-i', '--initial_design',
                                default="presentation",
                                choices=['random', 'uniform', 'presentation'],
                                help='How to choose first observations.')
    cmdline_parser.add_argument('-v', '--verbose',
                                default=False,
                                help='verbosity',
                                action='store_true')
    cmdline_parser.add_argument('-s', '--seed',
                                default=15,
                                help='Which seed to use',
                                required=False,
                                type=int)
    cmdline_parser.add_argument('-p', '--print',
                                default=False,
                                help='Print graphs to file instead of displaying on screen.',
                                action='store_true')

    args, unknowns = cmdline_parser.parse_known_args()
    log_lvl = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_lvl)

    if unknowns:
        logging.warning('Found unknown arguments!')
        logging.warning(str(unknowns))
        logging.warning('These will be ignored')

    # init_size = max(1, int(args.num_func_evals * args.fraction_init))
    # Seed the RNG to obtain reproducible results
    SEED = args.seed
    np.random.seed(SEED)

    TOGGLE_PRINT = args.print
    if TOGGLE_PRINT:
        boplot.enable_printing()
    else:
        boplot.enable_onscreen_display()

    #init_size = max(1, int(args.num_func_evals * args.fraction_init))

    main(
        init_size=args.init_db_size,
        initial_design=args.initial_design,
    )
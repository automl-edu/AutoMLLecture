import warnings

warnings.filterwarnings('ignore')
import argparse
import logging
import os.path
import numpy as np
from sklearn.neighbors import KernelDensity as kd
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import Matern

from matplotlib import pyplot as plt
from matplotlib import ticker as mtick

import bo_plot_utils as boplot
from bo_configurations import *

TOGGLE_PRINT = False
INIT_X_PRESENTATION = [2.5, 3.5, 5.5, 7, 9]
OUTPUT_DIR = os.path.abspath("./outputs/es")
GP_SAMPLE_COLOR_SEED = 1256
GP_SAMPLE_SEED = 65

labels["xlabel"] = "$\lambda$"
labels["ylabel"] = ""
# labels["ylabel"] = "$c(\lambda')$"
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
    # 5. Show PDF derived from the histogram at 10e9 samples
    # 6. Mark maximum of the PDF as next configuration to be evaluated

    # a. Plot GP
    # b. Sample GP, mark minima, update histogram of lambda*
    # c. Repeat 2 for each sample.
    # d. Show results after multiple iterations

    boplot.set_rc('figure', figsize=(22, 11))

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

    def bin_large_sample_size(nsamples, seed, return_pdf=False, batch_size=1280000):
        # Used for plotting a histogram when a large number of samples are to be generated.

        logging.info(f"Generating batch-wise histogram data for {nsamples} samples {batch_size} samples at at time.")

        rng = np.random.RandomState(seed=seed)
        counts = np.zeros_like(X_.flatten())
        bin_edges = np.zeros(shape=(counts.shape[0]+1))
        # Smoothen out the batches - we don't care about missing out a small overflow number of samples.
        nsamples = (nsamples // batch_size) * batch_size
        for idx in range(0, nsamples, batch_size):
            # # Iterate in increments of batch_size samples, but check for an uneven batch in the last iteration
            # batch_nsamples = batch_size if (nsamples - idx) % batch_size == 0 else nsamples - idx
            if idx % (batch_size * 10) == 0:
                logging.info(f"Generated {idx} samples out of an expected {nsamples}"
                             f"[{idx * 100.0 / nsamples}%].")
            batch_nsamples = batch_size
            mu = gp.sample_y(X=X_, n_samples=batch_nsamples, random_state=rng)
            minima = X_[np.argmin(mu, axis=0), 0]
            hist, bin_edges = np.histogram(
                minima, bins=nbins,
                range=bin_range, density=return_pdf,
            )
            counts += hist

        logging.info(f"Finished generating {nsamples} samples.")
        return counts, bin_edges


    def draw_samples(nsamples, ax1, ax2, show_min=False, return_pdf=False, show_samples=True, show_hist=True, data=None):
        if not nsamples:
            raise RuntimeError(f"Number of samples must be a positive integer, received "
                               f"{nsamples} of type {type(nsamples)}")

        # If data is not None, assume that it contains pre-computed histogram data
        logging.debug("Recieved histogram data of shape %s." % str(np.array(data).shape))
        if data:
            logging.debug("Histogram data contained %d counts and %d bins." %(np.array(data[0]).shape[0], np.array(data[1]).shape[0]))
            counts = data[0]
            bins = data[1]
            return ax2.hist(
                bins[:-1],
                bins=bins,
                density=return_pdf,
                weights=counts,
                color='lightgreen', edgecolor='black', alpha=0.0 if return_pdf else 1.0
            )

        mu = gp.sample_y(X=X_, n_samples=nsamples, random_state=GP_SAMPLE_SEED)
        if show_samples:
            boplot.plot_gp_samples(
                mu=mu,
                nsamples=nsamples,
                precision=histogram_precision,
                custom_x=X_,
                show_min=show_min,
                ax=ax1,
                seed=GP_SAMPLE_COLOR_SEED
            )
        minima = X_[np.argmin(mu, axis=0), 0]
        logging.info("Shape of minima is {}".format(minima.shape))
        # logging.debug("minima is: {}".format(minima))

        bins = None
        if show_hist:
            bins = ax2.hist(
                minima, bins=nbins,
                range=bin_range, density=return_pdf,
                color='lightgreen', edgecolor='black', alpha=0.0 if return_pdf else 1.0
            )

        return bins


    def draw_basic_plot(ax2_sci_not=False):
        fig, (ax1, ax2) = plt.subplots(2, 1, squeeze=True)
        ax1.set_xlim(bounds['x'])
        ax1.set_ylim(bounds['gp_y'])
        ax2.set_xlim(bounds['x'])
        ax2.set_ylim(bounds['acq_y'])

        if ax2_sci_not:
            f = mtick.ScalarFormatter(useOffset=False, useMathText=True)
            g = lambda x,pos : "${}$".format(f._formatSciNotation('%1.10e' % x))
            ax2.yaxis.set_major_formatter(mtick.FuncFormatter(g))

        ax1.grid()
        ax2.grid()

        boplot.plot_objective_function(ax=ax1)
        boplot.mark_observations(X_=x, Y_=y, mark_incumbent=False, ax=ax1)

        return fig, (ax1, ax2)


    def draw_freq_plots(nsamples):
        fig, (ax1, ax2) = draw_basic_plot()

        if nsamples == 1:
            show_min = True
        else:
            show_min = False

        draw_samples(nsamples=nsamples, ax1=ax1, ax2=ax2, show_min=show_min)

        return fig, (ax1, ax2)


    def finishing_touches(ax1, ax2, ax1_title, ax2_title, show_legend=False, figname="es.pdf"):
        ax1.set_xlabel(labels['xlabel'])
        # ax1.set_ylabel(labels['gp_ylabel'])
        # ax1.set_title(ax1_title, loc='left')

        ax2.set_xlabel(labels['xlabel'])
        ax2.set_ylabel(r'Frequency')
        ax2.set_title(ax2_title, loc='left')

        if show_legend:
            ax1.legend().set_zorder(zorders["legend"])
        else:
            ax1.legend().remove()

        # plt.tight_layout()
        plt.subplots_adjust(hspace=1.0)
        if TOGGLE_PRINT:
            plt.savefig(f"{OUTPUT_DIR}/{figname}")
        else:
            plt.show()

    # 1. Show GP fit on initial dataset, 0 samples, histogram
    # -------------------------------------------

    ax2_title = r'$p_{min}=P(\lambda=\lambda^*)$'

    bounds['acq_y'] = (0.0, 1.0)

    fig, (ax1, ax2) = draw_basic_plot()
    boplot.plot_gp(model=gp, confidence_intervals=[1.0, 2.0, 3.0], ax=ax1, custom_x=x)

    # Plot uniform prior for p_min
    xplot = boplot.get_plot_domain()
    ylims = ax2.get_ylim()
    xlims = ax2.get_xlim()
    yupper = [(ylims[1] - ylims[0]) / (xlims[1] - xlims[0])] * xplot.shape[0]
    ax2.plot(xplot[:, 0], yupper, color='green', linewidth=2.0)
    ax2.fill_between(xplot[:, 0], ylims[0], yupper, color='lightgreen')

    # ax1.legend().set_zorder(zorders["legend"])
    ax1.set_xlabel(labels['xlabel'])
    # ax1.set_ylabel(labels['gp_ylabel'])
    # ax1.set_title(r"Visualization of $\mathcal{G}^t$", loc='left')

    ax2.set_xlabel(labels['xlabel'])
    ax2.set_ylabel(r'$p_{min}$')
    ax2.set_title(ax2_title, loc='left')

    plt.tight_layout()
    if TOGGLE_PRINT:
        plt.savefig(f"{OUTPUT_DIR}/es_1.pdf")
    else:
        plt.show()
    # -------------------------------------------


    # 2. Show GP fit on initial dataset, 1 sample, histogram
    # -------------------------------------------

    nsamples = 1
    bounds['acq_y'] = (0.0, 5.0)
    ax1_title = r"One sample$"
    ax2_title = r'Frequency of $\lambda=\hat{\lambda}^*$'
    figname = "es_2.pdf"

    fig, (ax1, ax2) = draw_freq_plots(nsamples=nsamples)

    finishing_touches(
        ax1=ax1, ax2=ax2,
        ax1_title=ax1_title, ax2_title=ax2_title,
        show_legend=False,
        figname=figname
    )

    # 3. Show GP fit on initial dataset, 10 samples, histogram
    # -------------------------------------------

    nsamples = 10
    bounds['acq_y'] = (0.0, 10.0)
    ax1_title = r"Ten samples$"
    ax2_title = r'Frequency of $\lambda=\hat{\lambda}^*$'
    figname = "es_3.pdf"

    fig, (ax1, ax2) = draw_freq_plots(nsamples=nsamples)

    finishing_touches(
        ax1=ax1, ax2=ax2,
        ax1_title=ax1_title, ax2_title=ax2_title,
        show_legend=False,
        figname=figname
    )

    # -------------------------------------------

    # 4. Show GP fit on initial dataset, 200 samples, histogram
    # -------------------------------------------

    nsamples = 100
    bounds["acq_y"] = (0.0, 20.0)
    ax1_title = r"200 samples$"
    ax2_title = r'Frequency of $\lambda=\hat{\lambda}^*$'
    figname="es_4.pdf"

    fig, (ax1, ax2) = draw_freq_plots(nsamples=nsamples)

    finishing_touches(
        ax1=ax1, ax2=ax2,
        ax1_title=ax1_title, ax2_title=ax2_title,
        show_legend=False,
        figname=figname
    )

    # -------------------------------------------

    # 5. Show PDF derived from the histogram at 10e9 samples
    # -------------------------------------------

    nsamples = int(1e9)    # Generate ~1 Billion samples
    bounds["acq_y"] = (0.0, nsamples / 10.0)
    # ax1_title = r"200 samples from $\mathcal{G}^t$"
    # ax2_title = "$\hat{P}(\lambda=\lambda^*)$"
    ax1_title = r"A very large number of samples"
    ax2_title = r'Frequency of $\lambda=\hat{\lambda}^*$'
    figname = "es_5.pdf"

    fig, (ax1, ax2) = draw_basic_plot(ax2_sci_not=True)

    # Draw only a limited number of samples
    draw_samples(nsamples=200, ax1=ax1, ax2=ax2, show_min=False, show_samples=True, show_hist=False)

    # Use an alternate procedure to generate the histogram data
    counts, bins = bin_large_sample_size(nsamples, seed=GP_SAMPLE_SEED, return_pdf=False, batch_size=1280000)
    hist_data = (counts, bins)

    # Draw histogram only for a large number of samples
    draw_samples(nsamples=nsamples, ax1=ax1, ax2=ax2, show_min=False,
                 show_samples=False, show_hist=True, data=hist_data)

    finishing_touches(
        ax1=ax1, ax2=ax2,
        ax1_title=ax1_title, ax2_title=ax2_title,
        show_legend=False,
        figname=figname
    )

    # -------------------------------------------

    # 6. Mark maximum of the PDF as next configuration to be evaluated
    # -------------------------------------------

    figname = "es_6.pdf"

    fig, (ax1, ax2) = draw_basic_plot(ax2_sci_not=True)

    # Draw only a limited number of samples
    draw_samples(nsamples=200, ax1=ax1, ax2=ax2, show_min=False, show_samples=True, show_hist=False)

    # Draw histogram only for a large number of samples using previously generated histogram data
    draw_samples(nsamples=nsamples, ax1=ax1, ax2=ax2,
                        show_min=False, show_samples=False, show_hist=True, data=hist_data)

    xplot = boplot.get_plot_domain()

    idx_umax = np.argmax(counts)
    xmax = (bins[idx_umax] + bins[idx_umax + 1]) / 2.0
    logging.info(f"Highlighting xmax as configuration at index {idx_umax} with count {counts[idx_umax]}, "
                 f"at configuration {xmax}.")
    boplot.highlight_configuration(x=xmax, label=r'$\lambda^{(t)}$', ax=ax1, disable_ticks=False)
    # boplot.annotate_x_edge(label=r'$\lambda^{(t)}$', xy=(xplot[idx_umax], ax1.get_ylim()[0]),
    #                        ax=ax1, align='top', offset_param=1.5)
    boplot.highlight_configuration(x=xmax, label=r'$\lambda^{(t)}$', ax=ax2, disable_ticks=False)
    # boplot.annotate_x_edge(label=r'$\lambda^{(t)}$', xy=(xplot[idx_umax], ys[idx_umax]),
    #                        ax=ax2, align='top', offset_param=1.0)

    finishing_touches(
        ax1=ax1, ax2=ax2,
        ax1_title=ax1_title, ax2_title=ax2_title,
        show_legend=False,
        figname=figname
    )

    # nsamples = 200
    # seed3 = 65
    #
    # mu = gp.sample_y(X=X_, n_samples=nsamples, random_state=seed3)
    # data_h = X_[np.argmin(mu, axis=0), 0]
    #
    # kde = kd(kernel='gaussian', bandwidth=0.75).fit(data_h.reshape(-1, 1))
    #
    # ax2.plot(xplot, ys, color='green', lw=2.)
    # ax2.fill_between(xplot[:, 0], ax2.get_ylim()[0], ys, color='lightgreen')

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

    # init_size = max(1, int(args.num_func_evals * args.fraction_init))

    main(
        init_size=args.init_db_size,
        initial_design=args.initial_design,
    )

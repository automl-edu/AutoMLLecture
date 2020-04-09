import warnings
warnings.filterwarnings('ignore')
import argparse
import logging
from functools import partial

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import Matern

from matplotlib import pyplot as plt

import bo_plot_utils as boplot
from bo_configurations import *


SEED = None
TOGGLE_PRINT = False
INIT_X_PRESENTATION = [2.5, 3.5, 5.5, 7, 9]
bounds["x"] = (2, 13)
bounds["gp_y"] = (-5, 5)
# boplot.set_rcparams(**{"legend.loc": "lower left"})

labels["xlabel"] = "$\lambda'$"
labels["gp_ylabel"] = "$c(\lambda')$"

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
        x = np.linspace(bounds["x"][0], bounds["x"][1], init).reshape(-1, 1).tolist()
    elif initial_design == 'random':
        x = np.random.uniform(bounds["x"][0], bounds["x"][1], init).reshape(-1, 1).tolist()
    elif initial_design == 'presentation':
        x = np.array(INIT_X_PRESENTATION).reshape(-1, 1).tolist()

    # get corresponding response values
    y = list(map(f, x))

    return x, y


def get_lcb_maximum(model, kappa):
    """
    Given a model, return the (x, y) coords of LCB maximum.
    :param model: The underlying GP.
    :return:
    """

    X_ = boplot.get_plot_domain()
    mu, sigma = model.predict(X_, return_std=True)

    lcb = - (mu - kappa * sigma).reshape(-1)

    idx = np.argmax(lcb)
    logging.info("Found lcb maximum at index {}:{}".format(idx, (X_[idx, 0], lcb[idx])))
    return (X_[idx, 0], -lcb[idx])


def visualize_lcb(initial_design, init=None):
    """
    Visualize one-step of LCB.
    :param initial_design: Method for initializing the GP, choice between 'uniform', 'random', and 'presentation'
    :param init: Number of datapoints to initialize GP with.
    :return: None
    """

    # 1. Plot 3 different confidence envelopes
    # 2. Plot only LCB for one confidence envelope
    # 3. Mark next sample

    boplot.set_rcparams(**{'legend.loc': 'lower right'})

    logging.debug("Visualizing PI with initial design {} and init {}".format(initial_design, init))
    # Initialize dummy dataset
    x, y = initialize_dataset(initial_design=initial_design, init=init)
    logging.debug("Initialized dataset with:\nsamples {0}\nObservations {1}".format(x, y))

    # Fit GP to the currently available dataset
    gp = GPR(kernel=Matern())
    logging.debug("Fitting GP to\nx: {}\ny:{}".format(x, y))
    gp.fit(x, y)  # fit the model

    # noinspection PyStringFormat
    logging.debug("Model fit to dataset.\nOriginal Inputs: {0}\nOriginal Observations: {1}\n"
                  "Predicted Means: {2}\nPredicted STDs: {3}".format(x, y, *(gp.predict(x, return_std=True))))

    # 1. Plot 3 different confidence envelopes
    # -------------Plotting code -----------------

    fig, ax = plt.subplots(1, 1, squeeze=True)
    ax.set_xlim(bounds["x"])
    ax.set_ylim(bounds["gp_y"])
    ax.grid()
    boplot.plot_gp(model=gp, confidence_intervals=[1.0, 2.0, 3.0], type='both', custom_x=x, ax=ax)
    boplot.plot_objective_function(ax=ax)
    boplot.mark_observations(X_=x, Y_=y, mark_incumbent=False, highlight_datapoint=None, highlight_label=None, ax=ax)

    ax.legend().set_zorder(20)
    ax.set_xlabel(labels['xlabel'])
    ax.set_ylabel(labels['gp_ylabel'])
    ax.set_title(r"Visualization of $\mathcal{G}^{(t)}$", loc='left')

    plt.tight_layout()
    if TOGGLE_PRINT:
        plt.savefig("lcb_1.pdf")
    else:
        plt.show()
    # -------------------------------------------

    # 2. Plot only LCB for one confidence envelope
    # -------------Plotting code -----------------

    boplot.set_rcparams(**{'legend.loc': 'upper left'})
    kappa = 3.0

    fig, ax = plt.subplots(1, 1, squeeze=True)
    ax.set_xlim(bounds["x"])
    ax.set_ylim(bounds["gp_y"])
    ax.grid()
    boplot.plot_gp(model=gp, confidence_intervals=[kappa], type='lower', custom_x=x, ax=ax)
    boplot.plot_objective_function(ax=ax)
    boplot.mark_observations(X_=x, Y_=y, mark_incumbent=True, highlight_datapoint=None, highlight_label=None, ax=ax)

    ax.legend().set_zorder(20)
    ax.set_xlabel(labels['xlabel'])
    ax.set_ylabel(labels['gp_ylabel'])
    ax.set_title(r"Visualization of $\mathcal{G}^{(t)}$", loc='left')

    plt.tight_layout()
    if TOGGLE_PRINT:
        plt.savefig("lcb_2.pdf")
    else:
        plt.show()
    # -------------------------------------------

    # 3. Show LCB in parallel
    # -------------Plotting code -----------------

    if TOGGLE_PRINT:
        fig, (ax1, ax2) = plt.subplots(2, 1, squeeze=True, figsize=(18, 9))
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, squeeze=True)

    ax1.set_xlim(bounds["x"])
    ax1.set_ylim(bounds["gp_y"])
    ax1.grid()
    boplot.plot_gp(model=gp, confidence_intervals=[kappa], type='lower', custom_x=x, ax=ax1)
    boplot.plot_objective_function(ax=ax1)
    boplot.mark_observations(X_=x, Y_=y, mark_incumbent=True, highlight_datapoint=None, highlight_label=None, ax=ax1)

    lcb_max = get_lcb_maximum(gp, kappa)
    logging.info("LCB Maximum at:{}".format(lcb_max))
    boplot.highlight_configuration(x=lcb_max[0], label=None, lloc='bottom', ax=ax1)
    ax1.set_xlabel(labels['xlabel'])
    ax1.set_ylabel(labels['gp_ylabel'])
    ax1.set_title(r"Visualization of $\mathcal{G}^{(t)}$", loc='left')

    ax2.set_xlim(bounds["x"])
    ax2.set_ylim(bounds["acq_y"])
    ax2.grid()
    ax2.set_xlabel(labels['xlabel'])
    ax2.set_ylabel(labels['acq_ylabel'])
    ax2.set_title(r"Visualization of $LCB$", loc='left')

    boplot.highlight_configuration(x=lcb_max[0], label=None, lloc='bottom', ax=ax2)
    boplot.plot_acquisition_function(acquisition_functions['LCB'], 0.0, gp, kappa, ax=ax2)

    plt.tight_layout()
    if TOGGLE_PRINT:
        plt.savefig("lcb_3.pdf")
    else:
        plt.show()
    # -------------------------------------------

    # 4. Mark next sample
    # -------------Plotting code -----------------
    fig, ax = plt.subplots(1, 1, squeeze=True)
    ax.set_xlim(bounds["x"])
    ax.set_ylim(bounds["gp_y"])
    ax.grid()
    boplot.plot_gp(model=gp, confidence_intervals=[3.0], type='lower', custom_x=x, ax=ax)
    boplot.plot_objective_function(ax=ax)
    boplot.mark_observations(X_=x, Y_=y, mark_incumbent=True, highlight_datapoint=None, highlight_label=None, ax=ax)

    lcb_max = get_lcb_maximum(gp, 3.0)
    logging.info("LCB Maximum at:{}".format(lcb_max))
    boplot.highlight_configuration(x=lcb_max[0], label=None, lloc='bottom', ax=ax)
    boplot.highlight_output(y=lcb_max[1], label='', lloc='left', ax=ax)
    boplot.annotate_y_edge(label=r'${\hat{c}}^{(t)}(%.2f)$' % lcb_max[0], xy=lcb_max, align='left', ax=ax)

    ax.legend().set_zorder(20)
    ax.set_xlabel(labels['xlabel'])
    ax.set_ylabel(labels['gp_ylabel'])
    ax.set_title(r"Visualization of $\mathcal{G}^{(t)}$", loc='left')

    ax.legend().remove()

    plt.tight_layout()
    if TOGGLE_PRINT:
        plt.savefig("lcb_4.pdf")
    else:
        plt.show()
    # -------------------------------------------


def main(init_size, initial_design):
    visualize_lcb(
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
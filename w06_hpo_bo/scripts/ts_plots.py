import warnings
warnings.filterwarnings('ignore')
import argparse
import logging
from functools import partial
import os.path
import numpy as np
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import Matern

from matplotlib import pyplot as plt

import bo_plot_utils as boplot
from bo_configurations import *

SEED = None
TOGGLE_PRINT = False
INIT_X_PRESENTATION = [2.5, 3.5, 5.5, 7, 9]
OUTPUT_DIR = os.path.abspath("./outputs/ts")
bounds["x"] = (2, 13)
bounds["gp_y"] = (-5, 5)

labels["xlabel"] = "$\lambda$"
labels["gp_ylabel"] = ""

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


def visualize_ts(initial_design, init=None):
    """
    Visualize one-step of TS.
    :param initial_design: Method for initializing the GP, choice between 'uniform', 'random', and 'presentation'
    :param init: Number of datapoints to initialize GP with.
    :return: None
    """

    # 1. Plot GP fit on initial dataset
    # 2. Plot one sample
    # 3. Mark minimum of sample

    boplot.set_rcparams(**{'legend.loc': 'upper left'})

    logging.debug("Visualizing EI with initial design {} and init {}".format(initial_design, init))
    # Initialize dummy dataset
    x, y = initialize_dataset(initial_design=initial_design, init=init)
    ymin_arg = np.argmin(y)
    ymin = y[ymin_arg]
    logging.debug("Initialized dataset with:\nsamples {0}\nObservations {1}".format(x, y))

    # Fit GP to the currently available dataset
    gp = GPR(kernel=Matern())
    logging.debug("Fitting GP to\nx: {}\ny:{}".format(x, y))
    gp.fit(x, y)  # fit the model

    # noinspection PyStringFormat
    logging.debug("Model fit to dataset.\nOriginal Inputs: {0}\nOriginal Observations: {1}\n"
                  "Predicted Means: {2}\nPredicted STDs: {3}".format(x, y, *(gp.predict(x, return_std=True))))



    # 1. Plot GP fit on initial dataset
    # -------------Plotting code -----------------
    fig, ax = plt.subplots(1, 1, squeeze=True)
    ax.set_xlim(bounds["x"])
    ax.set_ylim(bounds["gp_y"])
    ax.grid()
    boplot.plot_gp(model=gp, confidence_intervals=[1.0, 2.0, 3.0], custom_x=x, ax=ax)
    boplot.plot_objective_function(ax=ax)
    boplot.mark_observations(X_=x, Y_=y, mark_incumbent=False, highlight_datapoint=None, highlight_label=None, ax=ax)

    ax.legend().set_zorder(zorders['legend'])
    ax.set_xlabel(labels['xlabel'])

    plt.tight_layout()
    if TOGGLE_PRINT:
        plt.savefig(f"{OUTPUT_DIR}/ts_1.pdf")
    else:
        plt.show()
    # -------------------------------------------

    # 2. Plot one sample
    # -------------Plotting code -----------------
    fig, ax = plt.subplots(1, 1, squeeze=True)
    ax.set_xlim(bounds["x"])
    ax.set_ylim(bounds["gp_y"])
    ax.grid()
    boplot.plot_gp(model=gp, confidence_intervals=[1.0, 2.0, 3.0], custom_x=x, ax=ax)
    boplot.plot_objective_function(ax=ax)
    boplot.mark_observations(X_=x, Y_=y, mark_incumbent=False, highlight_datapoint=None, highlight_label=None, ax=ax)

    # Sample from the GP
    nsamples = 1
    seed2 = 2
    seed3 = 1375
    X_ = boplot.get_plot_domain(precision=None)
    mu = gp.sample_y(X=X_, n_samples=nsamples, random_state=seed3)
    boplot.plot_gp_samples(
        mu=mu,
        nsamples=nsamples,
        precision=None,
        custom_x=X_,
        show_min=False,
        ax=ax,
        seed=seed2
    )

    ax.legend().set_zorder(zorders['legend'])
    ax.set_xlabel(labels['xlabel'])

    plt.tight_layout()
    if TOGGLE_PRINT:
        plt.savefig(f"{OUTPUT_DIR}/ts_2.pdf")
    else:
        plt.show()
    # -------------------------------------------

    # 3. Mark minimum of sample
    # -------------Plotting code -----------------
    fig, ax = plt.subplots(1, 1, squeeze=True)
    ax.set_xlim(bounds["x"])
    ax.set_ylim(bounds["gp_y"])
    ax.grid()
    # boplot.plot_gp(model=gp, confidence_intervals=[2.0], custom_x=x, ax=ax)
    boplot.plot_objective_function(ax=ax)
    boplot.mark_observations(X_=x, Y_=y, mark_incumbent=False, highlight_datapoint=None, highlight_label=None, ax=ax)

    # Sample from the GP
    nsamples = 1
    X_ = boplot.get_plot_domain(precision=None)
    mu = gp.sample_y(X=X_, n_samples=nsamples, random_state=seed3)
    colors['highlighted_observations'] = 'red'  # Special for Thompson Sampling
    boplot.plot_gp_samples(
        mu=mu,
        nsamples=nsamples,
        precision=None,
        custom_x=X_,
        show_min=True,
        ax=ax,
        seed=seed2
    )

    min_idx = np.argmin(mu, axis=0)
    candidate = X_[min_idx]
    cost = mu[min_idx]

    boplot.highlight_configuration(x=np.array([candidate]), label='', lloc='bottom', disable_ticks=True, ax=ax)
    boplot.annotate_x_edge(label=r'$\lambda^{(t)}$',xy=(candidate, cost), align='top', ax=ax)
    boplot.highlight_output(y=np.array([cost]), label='', lloc='left', disable_ticks=True, ax=ax)
    boplot.annotate_y_edge(label=r'$g(\lambda^{(t)})$', xy=(candidate, cost), align='left', ax=ax)

    ax.legend().set_zorder(zorders['legend'])
    ax.set_xlabel(labels['xlabel'])

    plt.tight_layout()
    if TOGGLE_PRINT:
        plt.savefig(f"{OUTPUT_DIR}/ts_3.pdf")
    else:
        plt.show()
    # -------------------------------------------


def main(init_size, initial_design):
        visualize_ts(
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
import warnings
warnings.filterwarnings('ignore')
import argparse
import logging
from functools import partial

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


def visualize_ei(initial_design, init=None):
    """
    Visualize one-step of EI.
    :param initial_design: Method for initializing the GP, choice between 'uniform', 'random', and 'presentation'
    :param init: Number of datapoints to initialize GP with.
    :return: None
    """

    # 1. Plot GP fit on initial dataset
    # 2. Mark current incumbent
    # 3. Mark Zone of Probable Improvement
    # 4. Mark Hypothetical Real cost of a random configuration
    # 5. Display I(lambda)
    # 6. Display Vertical Normal Distribution

    # boplot.set_rcparams(**{'legend.loc': 'lower left'})

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
    mu_star_t_xy = get_mu_star(gp)
    logging.info("Mu-star at time t: {}".format(mu_star_t_xy))

    # noinspection PyStringFormat
    logging.debug("Model fit to dataset.\nOriginal Inputs: {0}\nOriginal Observations: {1}\n"
                  "Predicted Means: {2}\nPredicted STDs: {3}".format(x, y, *(gp.predict(x, return_std=True))))

    # 1. Plot GP fit on initial dataset
    # -------------Plotting code -----------------
    fig, ax = plt.subplots(1, 1, squeeze=True)
    ax.set_xlim(bounds["x"])
    ax.set_ylim(bounds["gp_y"])
    ax.grid()
    boplot.plot_gp(model=gp, confidence_intervals=[2.0], custom_x=x, ax=ax)
    boplot.plot_objective_function(ax=ax)
    boplot.mark_observations(X_=x, Y_=y, mark_incumbent=False, highlight_datapoint=None, highlight_label=None, ax=ax)

    ax.legend().set_zorder(20)
    ax.set_xlabel(labels['xlabel'])
    ax.set_ylabel(labels['gp_ylabel'])
    ax.set_title(r"Visualization of $\mathcal{G}^{(t)}$", loc='left')

    plt.tight_layout()
    if TOGGLE_PRINT:
        plt.savefig("ei_1.pdf")
    else:
        plt.show()
    # -------------------------------------------

    # 2. Mark current incumbent
    # -------------Plotting code -----------------
    fig, ax = plt.subplots(1, 1, squeeze=True)
    ax.set_xlim(bounds["x"])
    ax.set_ylim(bounds["gp_y"])
    ax.grid()
    boplot.plot_gp(model=gp, confidence_intervals=[2.0], custom_x=x, ax=ax)
    boplot.plot_objective_function(ax=ax)
    boplot.mark_observations(X_=x, Y_=y, mark_incumbent=True, highlight_datapoint=None, highlight_label=None, ax=ax)

    ax.legend().set_zorder(20)
    ax.set_xlabel(labels['xlabel'])
    ax.set_ylabel(labels['gp_ylabel'])
    ax.set_title(r"Visualization of $\mathcal{G}^{(t)}$", loc='left')

    plt.tight_layout()
    if TOGGLE_PRINT:
        plt.savefig("ei_2.pdf")
    else:
        plt.show()
    # -------------------------------------------

    # 3. Mark Zone of Probable Improvement
    # -------------Plotting code -----------------
    fig, ax = plt.subplots(1, 1, squeeze=True)
    ax.set_xlim(bounds["x"])
    ax.set_ylim(bounds["gp_y"])
    ax.grid()
    boplot.plot_gp(model=gp, confidence_intervals=[2.0], custom_x=x, ax=ax)
    boplot.plot_objective_function(ax=ax)
    boplot.mark_observations(X_=x, Y_=y, mark_incumbent=True, highlight_datapoint=None, highlight_label=None, ax=ax)
    boplot.darken_graph(y=ymin, ax=ax)

    ax.legend().set_zorder(20)
    ax.set_xlabel(labels['xlabel'])
    ax.set_ylabel(labels['gp_ylabel'])
    ax.set_title(r"Visualization of $\mathcal{G}^{(t)}$", loc='left')

    ax.legend().remove()

    plt.tight_layout()
    if TOGGLE_PRINT:
        plt.savefig("ei_3.pdf")
    else:
        plt.show()
    # -------------------------------------------

    # 4. Forget the underlying objective function
    # -------------------------------------------

    fig, ax = plt.subplots(1, 1, squeeze=True)
    ax.set_xlim(bounds["x"])
    ax.set_ylim(bounds["gp_y"])
    ax.grid()
    boplot.plot_gp(model=gp, confidence_intervals=[2.0], custom_x=x, ax=ax)
    # boplot.plot_objective_function(ax=ax)
    boplot.mark_observations(X_=x, Y_=y, mark_incumbent=True, highlight_datapoint=None, highlight_label=None, ax=ax)
    boplot.darken_graph(y=ymin, ax=ax)

    ax.set_xlabel(labels['xlabel'])
    ax.set_ylabel(labels['gp_ylabel'])
    ax.set_title(r"Visualization of $\mathcal{G}^{(t)}$", loc='left')

    plt.tight_layout()
    if TOGGLE_PRINT:
        plt.savefig("ei_4.pdf")
    else:
        plt.show()

    # -------------------------------------------

    # 5. Mark Hypothetical Real cost of a random configuration
    # -------------------------------------------

    fig, ax = plt.subplots(1, 1, squeeze=True)
    ax.set_xlim(bounds["x"])
    ax.set_ylim(bounds["gp_y"])
    ax.grid()
    boplot.plot_gp(model=gp, confidence_intervals=[2.0], custom_x=x, ax=ax)
    # boplot.plot_objective_function(ax=ax)
    boplot.mark_observations(X_=x, Y_=y, mark_incumbent=True, highlight_datapoint=None, highlight_label=None, ax=ax)
    boplot.darken_graph(y=ymin, ax=ax)

    ax.legend().set_zorder(20)
    ax.set_xlabel(labels['xlabel'])
    ax.set_ylabel(labels['gp_ylabel'])
    ax.set_title(r"Visualization of $\mathcal{G}^{(t)}$", loc='left')

    candidate = 11.0
    cost = -3.5
    boplot.highlight_configuration(x=np.array([candidate]), label=r'$\lambda$', lloc='bottom', ax=ax)
    boplot.highlight_output(y=np.array([cost]), label='', lloc='left', ax=ax)
    boplot.annotate_y_edge(label=r'$c(\lambda)$', xy=(candidate, cost), align='left', ax=ax)

    ax.legend().remove()

    plt.tight_layout()
    if TOGGLE_PRINT:
        plt.savefig("ei_5.pdf")
    else:
        plt.show()

    # -------------------------------------------

    # 6. Display I(lambda)
    # -------------------------------------------

    fig, ax = plt.subplots(1, 1, squeeze=True)
    ax.set_xlim(bounds["x"])
    ax.set_ylim(bounds["gp_y"])
    ax.grid()
    boplot.plot_gp(model=gp, confidence_intervals=[2.0], custom_x=x, ax=ax)
    # boplot.plot_objective_function(ax=ax)
    boplot.mark_observations(X_=x, Y_=y, mark_incumbent=True, highlight_datapoint=None, highlight_label=None, ax=ax)
    boplot.darken_graph(y=ymin, ax=ax)

    ax.legend().set_zorder(20)
    ax.set_xlabel(labels['xlabel'])
    ax.set_ylabel(labels['gp_ylabel'])
    ax.set_title(r"Visualization of $\mathcal{G}^{(t)}$", loc='left')

    boplot.highlight_configuration(x=np.array([candidate]), label=r'$\lambda$', lloc='bottom', ax=ax)
    boplot.highlight_output(y=np.array([cost]), label='', lloc='left', ax=ax)
    boplot.annotate_y_edge(label=r'$c(\lambda)$', xy=(candidate, cost), align='left', ax=ax)

    xmid = (x[ymin_arg][0] + candidate) / 2.
    ax.annotate(s='', xy=(xmid, cost), xytext=(xmid, ymin),
                 arrowprops={'arrowstyle': '<|-|>',})

    textx = xmid + (ax.get_xlim()[1] - ax.get_xlim()[0]) / 40
    ax.text(textx, (ymin + cost) / 2, r'$I^{(t)}(\lambda)$', weight='heavy')

    ax.legend().remove()

    plt.tight_layout()
    if TOGGLE_PRINT:
        plt.savefig("ei_6.pdf")
    else:
        plt.show()

    # -------------------------------------------


    # 7. Display Vertical Normal Distribution
    # -------------------------------------------

    fig, ax = plt.subplots(1, 1, squeeze=True)
    ax.set_xlim(bounds["x"])
    ax.set_ylim(bounds["gp_y"])
    ax.grid()
    boplot.plot_gp(model=gp, confidence_intervals=[2.0], custom_x=x, ax=ax)
    # boplot.plot_objective_function(ax=ax)
    boplot.mark_observations(X_=x, Y_=y, mark_incumbent=True, highlight_datapoint=None, highlight_label=None, ax=ax)
    boplot.darken_graph(y=ymin, ax=ax)

    ax.legend().set_zorder(20)
    ax.set_xlabel(labels['xlabel'])
    ax.set_ylabel(labels['gp_ylabel'])
    ax.set_title(r"Visualization of $\mathcal{G}^{(t)}$", loc='left')

    boplot.highlight_configuration(x=np.array([candidate]), label=r'$\lambda$', lloc='bottom', ax=ax)
    boplot.highlight_output(y=np.array([cost]), label='', lloc='left', ax=ax)
    boplot.annotate_y_edge(label=r'$c(\lambda)$', xy=(candidate, cost), align='left', ax=ax)

    xmid = (x[ymin_arg][0] + candidate) / 2.
    ax.annotate(s='', xy=(xmid, cost), xytext=(xmid, ymin),
                 arrowprops={'arrowstyle': '<|-|>',})

    textx = xmid + (ax.get_xlim()[1] - ax.get_xlim()[0]) / 40
    ax.text(textx, (ymin + cost) / 2, r'$I^{(t)}(\lambda)$', weight='heavy')

    vcurve_x, vcurve_y, mu = boplot.draw_vertical_normal(
        gp=gp, incumbenty=ymin, ax=ax, xtest=candidate,
        xscale=2.0, yscale=1.0
    )

    ann_x = candidate + 0.3 * (np.max(vcurve_x) - candidate) / 2
    ann_y = ymin - (mu - ymin) / 2

    arrow_x = ann_x + 0.5
    arrow_y = ann_y - 3.0

    # label = "{:.2f}".format(candidate)
    label = '\lambda'

    ax.annotate(
        s=r'$PI^{(t)}(%s)$' % label, xy=(ann_x, ann_y), xytext=(arrow_x, arrow_y),
        arrowprops={'arrowstyle': 'fancy'},
        weight='heavy', color='darkgreen', zorder=15
    )

    ax.legend().remove()

    plt.tight_layout()
    if TOGGLE_PRINT:
        plt.savefig("ei_7.pdf")
    else:
        plt.show()

    # -------------------------------------------


def main(init_size, initial_design):
    visualize_ei(
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
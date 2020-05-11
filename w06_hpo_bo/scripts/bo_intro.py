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

import bo_intro_utils as boplot
from bo_configurations import *


SEED = None
INIT_X_PRESENTATION = [4.5, 10]
NUM_ACQ_OPTS = 50 # Number of times the acquisition function is optimized while looking for the next x to sample.
TOGGLE_PRINT = True


def initialize_dataset(initial_design, init=None):
    """
    Initialize some data to start fitting the GP on.
    :param initial_design: Method for initializing the GP, choice between 'uniform', 'random', and 'presentation'
    :param init: Number of datapoints to initialize with, if relevant
    :return:
    """

    # sample initial query points
    if initial_design == 'uniform':
        x = np.linspace(bounds["x_intro"][0], bounds["x_intro"][1], init).reshape(-1, 1).tolist()
    elif initial_design == 'random':
        x = np.random.uniform(bounds["x_intro"][0], bounds["x_intro"][1], init).reshape(-1, 1).tolist()
    elif initial_design == 'presentation':
        x = np.array(INIT_X_PRESENTATION).reshape(-1, 1).tolist()

    # get corresponding response values
    y = list(map(f, x))

    return x, y


def run_bo(acquisition, max_iter, initial_design, acq_add, init=None):
    """
    BO
    :param acquisition: type of acquisition function to be used
    :param max_iter: max number of function calls
    :param seed: seed used to keep experiments reproducible
    :param initial_design: Method for initializing the GP, choice between 'uniform', 'random', and 'presentation'
    :param acq_add: additional parameteres for acquisition function (e.g. kappa for LCB)
    :param init: Number of datapoints to initialize GP with.
    :return: all evaluated points.
    """

    logging.debug("Running BO with Acquisition Function {0}, maximum iterations {1}, initial design {2}, "
                  "acq_add {3} and init {4}".format(acquisition, max_iter, initial_design, acq_add, init))
    x, y = initialize_dataset(initial_design=initial_design, init=init)
    logging.debug("Initialized dataset with:\nsamples {0}\nObservations {1}".format(x, y))

    for i in range(1, max_iter):  # BO loop
        logging.debug('Sample #%d' % (i))

        # Fit GP to the currently available dataset
        gp = GPR(kernel=Matern())
        logging.debug("Fitting GP to\nx: {}\ny:{}".format(x, y))


        #Add data to fit a more stable GP
        x_add = np.add(x, 0.05)
        x_fit = np.append(x, x_add).reshape(-1, 1)
        y_fit = y + list(map(f, x_add))
        gp.fit(x_fit, y_fit)  # fit the model

        # ----------Plotting calls---------------
        fig, ax1 = plt.subplots(1, 1, squeeze=True)
        fig.tight_layout()
        ax1.set_xlim(bounds["x_intro"])
        ax1.set_ylim(bounds["y_intro"])
        ax1.set_yticks([])
        ax1.grid()

        boplot.plot_objective_function(ax=ax1, translation=10)

        annotate = False
        if i == 1:
             annotate = True

        boplot.plot_gp(model=gp, confidence_intervals=[1.0, 2.0, 3.0], ax=ax1, custom_x=x, annotate=annotate,
                       translation=10)

        mark_incumbent = False

        boplot.mark_observations(X_=x, Y_=y, ax=ax1, mark_incumbent=mark_incumbent, highlight_datapoint=len(y)-1)

        # # Partially initialize the acquisition function to work with the fmin interface
        # # (only the x parameter is not specified)
        acqui = partial(acquisition, model=gp, eta=min(y), add=acq_add)

        annotate = False
        if i == 1:
             annotate = True

        boplot.plot_acquisition_function(acquisition, min(y), gp, acq_add, invert=True, ax=ax1, annotate=annotate,
                                         scaling=30)

        # optimize acquisition function, repeat 10 times, use best result
        x_ = None
        y_ = 10000
        # Feel free to adjust the hyperparameters
        for j in range(NUM_ACQ_OPTS):
            opt_res = minimize(acqui, np.random.uniform(bounds["x_intro"][0], bounds["x_intro"][1]), method="L-BFGS-B", bounds=[(bounds["x_intro"][0], bounds["x_intro"][1])])
            if opt_res.fun[0] < y_:
                x_ = opt_res.x
                y_ = opt_res.fun[0]


        # Update dataset with new observation
        x.append(x_)
        y.append(f(x_))
        logging.info("After {0}. loop iteration".format(i))
        logging.info("x: {0:.3E}, y: {1:.3E}".format(x_[0], y_))

        if i==1:
             annotate_x = INIT_X_PRESENTATION[0]
             ax1.annotate("Observation", xy=(annotate_x, f([annotate_x])+ 10), xytext=(annotate_x - 1, f([annotate_x]) + 14),
                          arrowprops={'arrowstyle': 'fancy'}, zorder=10, fontsize='x-large')
             annotate_x = INIT_X_PRESENTATION[1]
             ax1.annotate("Objective function", xy=(annotate_x + 1, f([annotate_x + 1])+ 10), xytext=(annotate_x + 0.25, f([annotate_x + 1])+ 7),
                          arrowprops={'arrowstyle': 'fancy'}, zorder=10, fontsize='x-large')
        #
        if i==2:
             ax1.annotate("New observation", xy=(new_observation, f(new_observation)+ 10), xytext=(new_observation - 1, f(new_observation) + 6),
                          arrowprops={'arrowstyle': 'fancy'}, zorder=19, fontsize='x-large')

        ax1.set_xlabel(labels['xlabel'])

        new_observation = x_

        if TOGGLE_PRINT:
            plt.savefig("plot_{}.pdf".format(i), dpi='figure',bbox_inches = 'tight')
        else:
            plt.show()
        # ---------------------------------------

    return y


def main(num_evals, init_size, repetitions, initial_design, acq_add, acquisition):
    for i in range(repetitions):
        bo_res_1 = run_bo(max_iter=num_evals, init=init_size, initial_design=initial_design, acquisition=acquisition, acq_add=acq_add)


if __name__ == '__main__':
    cmdline_parser = argparse.ArgumentParser('AutoMLLecture')

    cmdline_parser.add_argument('-n', '--num_func_evals',
                                default=10,
                                help='Number of function evaluations',
                                type=int)
    cmdline_parser.add_argument('-f', '--init_db_size',
                                default=2,
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
    cmdline_parser.add_argument('-a', '--acquisition',
                                default='EI',
                                choices=['LCB', 'EI', 'PI'],
                                help='acquisition function')
    cmdline_parser.add_argument('-s', '--seed',
                                default=15,
                                help='Which seed to use',
                                required=False,
                                type=int)
    cmdline_parser.add_argument('-r', '--repetitions',
                                default=1,
                                help='Number of repeations for the experiment',
                                required=False,
                                type=int)
    cmdline_parser.add_argument('-p', '--print',
                                default=True,
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
        boplot.enable_printing(figsize=(30, 10))
    else:
        boplot.enable_onscreen_display()


    main(   num_evals=args.num_func_evals,
            init_size=args.init_db_size,
            repetitions=args.repetitions,
            initial_design=args.initial_design,
            acquisition=acquisition_functions[args.acquisition],
            acq_add=1
            )
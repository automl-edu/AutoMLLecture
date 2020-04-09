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
INIT_X_PRESENTATION = [3, 4, 4.6, 4.8, 5, 9.4, 10, 12.7]
NUM_ACQ_OPTS = 10 # Number of times the acquisition function is optimized while looking for the next x to sample.
TOGGLE_PRINT = False


def initialize_dataset(initial_design, init=None):
    """
    Initialize some data to start fitting the GP on.
    :param initial_design: Method for initializing the GP, choice between 'uniform', 'random', and 'presentation'
    :param init: Number of datapoints to initialize with, if relevant
    :return:
    """

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
        gp.fit(x, y)  # fit the model

        # ----------Plotting calls---------------
        fig, (ax1, ax2) = plt.subplots(2, 1, squeeze=True)
        fig.tight_layout()
        ax1.set_xlim(bounds["x"])
        ax1.set_ylim(bounds["gp_y"])
        ax1.grid()
        ax2.set_xlim(bounds["x"])
        ax2.set_ylim(bounds["acq_y"])
        ax2.grid()
        boplot.plot_objective_function(ax=ax1)
        boplot.plot_gp(model=gp, confidence_intervals=[1.0, 2.0], ax=ax1, custom_x=x)
        boplot.mark_observations(X_=x, Y_=y, ax=ax1)
        # ---------------------------------------

        # noinspection PyStringFormat
        logging.debug("Model fit to dataset.\nOriginal Inputs: {0}\nOriginal Observations: {1}\n"
                      "Predicted Means: {2}\nPredicted STDs: {3}".format(x, y, *(gp.predict(x, return_std=True))))

        # Partially initialize the acquisition function to work with the fmin interface
        # (only the x parameter is not specified)
        acqui = partial(acquisition, model=gp, eta=min(y), add=acq_add)

        boplot.plot_acquisition_function(acquisition, min(y), gp, acq_add, invert=True, ax=ax2)

        # optimize acquisition function, repeat 10 times, use best result
        x_ = None
        y_ = 10000
        # Feel free to adjust the hyperparameters
        for j in range(NUM_ACQ_OPTS):
            opt_res = minimize(acqui, np.random.uniform(bounds["x"][0], bounds["x"][1]),
                               #bounds=bounds["x"],
                               options={'maxfun': 20, 'maxiter': 20}, method="L-BFGS-B")
            if opt_res.fun[0] < y_:
                x_ = opt_res.x
                y_ = opt_res.fun[0]

        # ----------Plotting calls---------------
        boplot.highlight_configuration(x_, ax=ax1)
        boplot.highlight_configuration(x_, ax=ax2)
        # ---------------------------------------

        # Update dataset with new observation
        x.append(x_)
        y.append(f(x_))

        logging.info("After {0}. loop iteration".format(i))
        logging.info("x: {0:.3E}, y: {1:.3E}".format(x_[0], y_))



        # ----------Plotting calls---------------
        for ax in (ax1, ax2):
            ax.legend()
            ax.set_xlabel(labels['xlabel'])

        ax1.set_ylabel(labels['gp_ylabel'])
        ax1.set_title("Visualization of GP", loc='left')

        ax2.set_title("Visualization of Acquisition Function", loc='left')
        ax2.set_ylabel(labels['acq_ylabel'])
        if TOGGLE_PRINT:
            plt.savefig("plot_{}.pdf".format(i), dpi='figure')
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
                                default=5,
                                help='Number of function evaluations',
                                type=int)
    cmdline_parser.add_argument('-f', '--init_db_size',
                                default=4,
                                help='Size of the initial database',
                                type=int)
    cmdline_parser.add_argument('-i', '--initial_design',
                                default="random",
                                choices=['random', 'uniform', 'presentation'],
                                help='How to choose first observations.')
    cmdline_parser.add_argument('-v', '--verbose',
                                default=False,
                                help='verbosity',
                                action='store_true')
    cmdline_parser.add_argument('-a', '--acquisition',
                                default='LCB',
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

    main(   num_evals=args.num_func_evals,
            # init_size=init_size,
            init_size=args.init_db_size,
            repetitions=args.repetitions,
            initial_design=args.initial_design,
            acquisition=acquisition_functions[args.acquisition],
            # seed=args.seed,
            acq_add=1
            )
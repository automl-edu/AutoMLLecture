"""
Original file is located at
    https://colab.research.google.com/drive/1wAcalyfeaQFgy7Sg03WCB8Je5SozrxdE
"""

import warnings
warnings.filterwarnings('ignore')
import argparse
import logging
from functools import partial

import numpy as np
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import Matern
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from matplotlib import pyplot as plt

from bo_loop_acq_functions import EI, UCB
from bo_loop_utils import plot_search_graph, plot_acquisition_function, acquisition_functions
from bo_loop_obj_fun import f, bounds


def run_bo(acquisition, max_iter, init=2, seed=1, random=True, acq_add=1, mse_stop=0.001):
    """
    BO
    :param acquisition: type of acquisition function to be used
    :param max_iter: max number of function calls
    :param init: number of points to build initial model
    :param seed: seed used to keep experiments reproducible
    :param random: if False initial points are linearly sampled in the bounds, otherwise uniformly random
    :param acq_add: additional parameteres for acquisition function (e.g. kappa for UCB)
    :param mse_stop: assuming we know the optimum value in this toy problem, condtional to stop the search (Mean Square Error).
    :return: all evaluated points.
    """
    # sample initial query points
    np.random.seed(seed)
    if random:
        x = np.random.uniform(bounds['lower'], bounds['upper'], init).reshape(-1, 1).tolist()
    else:
        x = np.linspace(bounds['lower'], bounds['upper'], init).reshape(-1, 1).tolist()
    # get corresponding response values
    y = list(map(f, x))


    for i in range(max_iter - init):  # BO loop
        logging.debug('Sample #%d' % (init + i))
        #Feel free to adjust the hyperparameters
        gp = Pipeline([["standardize", StandardScaler()],
                      ["GP", GPR(kernel=Matern(nu=2.5), normalize_y=True, n_restarts_optimizer=10, random_state=seed)]])
        gp.fit(x, y)  # fit the model

        # Partially initialize the acquisition function to work with the fmin interface
        # (only the x parameter is not specified)
        acqui = partial(acquisition, model=gp, eta=min(y), add=acq_add)

        # optimize acquisition function, repeat 10 times, use best result
        x_ = None
        y_ = 10000
        # Feel free to adjust the hyperparameters
        for j in range(10):
            opt_res = minimize(acqui, np.random.uniform(bounds['lower'], bounds['upper']),
                               bounds=[[bounds['lower'], bounds['upper']]], options={'maxfun': 20, 'maxiter': 20}, method="L-BFGS-B")
            if opt_res.fun[0] < y_:
                x_ = opt_res.x
                y_ = opt_res.fun[0]

        x.append(x_)
        y.append(f(x_))

        print("After {0}. loop iteration".format(i))
        mse = mean_squared_error(x_, [1])
        print("x: {0:.3E}, y: {1:.3E}, MSE: {2:.3E}".format(x_[0], y_, mse))
        plot_search_graph(x, y, gp)
        plot_acquisition_function(acquisition, min(y), gp, acq_add)

        # assuming we know the optimum value, STOP when we are 'close enough'
        if mse < mse_stop:
            break

    return y



def main(num_evals, init_size, repetitions, random, seed, acq_add=1, acquisition=UCB):
    # Modify as you want
    for i in range(repetitions):
        bo_res_1 = run_bo(max_iter=num_evals, init=init_size, random=random, acquisition=acquisition, acq_add=acq_add, seed=seed+1)

if __name__ == '__main__':
    cmdline_parser = argparse.ArgumentParser('AutoMLLecture')

    cmdline_parser.add_argument('-n', '--num_func_evals',
                                default=10,
                                help='Number of function evaluations',
                                type=int)
    cmdline_parser.add_argument('-p', '--percentage_init',
                                default=0.4,
                                help='Percentage of budget (num_func_evals) to spend on building initial model',
                                type=float)
    cmdline_parser.add_argument('-r', '--random_initial_design',
                                action="store_true",
                                help='Use random initial points. If not set, initial points are sampled linearly on'
                                     ' the function bounds.')
    cmdline_parser.add_argument('-v', '--verbose',
                                default='INFO',
                                choices=['INFO', 'DEBUG'],
                                help='verbosity')
    cmdline_parser.add_argument('--acquisition',
                                default='UCB',
                                choices=['UCB', 'EI'],
                                help='acquisition function')
    cmdline_parser.add_argument('--seed',
                                default=14,
                                help='Which seed to use',
                                required=False,
                                type=int)
    cmdline_parser.add_argument('--repetitions',
                                default=1,
                                help='Number of repeations for the experiment',
                                required=False,
                                type=int)
    args, unknowns = cmdline_parser.parse_known_args()
    log_lvl = logging.INFO if args.verbose == 'INFO' else logging.DEBUG
    logging.basicConfig(level=log_lvl)

    if unknowns:
        logging.warning('Found unknown arguments!')
        logging.warning(str(unknowns))
        logging.warning('These will be ignored')

    init_size = max(1, int(args.num_func_evals * args.percentage_init))
    main(   num_evals=args.num_func_evals,
            init_size=init_size,
            repetitions=args.repetitions,
            random=args.random_initial_design,
            acquisition=acquisition_functions[args.acquisition],
            seed=args.seed
            )

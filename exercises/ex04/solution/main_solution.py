import argparse
import logging
from functools import partial

import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import Matern
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from matplotlib import pyplot as plt


def f(x):
    """
    Function to minimize. (Levy1D see https://www.sfu.ca/~ssurjano/levy.html). Global min value: 0.0
    """
    w0 = (1 + (x[0] - 1) / 4)
    term1 = np.power(np.sin(np.pi * w0), 2)

    term2 = 0
    for i in range(len(x) - 1):
        wi = 1 + (x[i] - 1) / 4
        term2 += np.power(wi - 1, 2) * (1 + 10 * np.power(np.sin(wi * np.pi + 1), 2))

    wd = (1 + (x[-1] - 1) / 4)
    term3 = np.power(wd - 1, 2)
    term3 *= (1 + np.power(np.sin(2 * np.pi * wd), 2))

    y = term1 + term2 + term3
    return y


def EI(x, model, eta, add=None):
    """
    Expected Improvement.
    :param x: point to determine the acquisition value
    :param model: GP to predict target function value
    :param eta: best so far seen value
    :param add: additional parameters necessary for the function
    """
    x = np.array([x]).reshape([-1, 1])
    m, s = model.predict(x, return_std=True)
    if s == 0: s = 1e-9
    d = norm(loc=m, scale=s)
    delta = eta-m
    ei = delta * d.cdf(1-delta/s) + s * d.pdf(delta/s)
    r = -ei
    return r

def UCB(x, model, eta, add=None):
    """
    Upper Confidence Bound
    :param x: point to determine the acquisition value
    :param model: GP to predict target function value
    :param add: additional parameters necessary for the function
    """
    x = np.array([x]).reshape([-1, 1])
    m, s = model.predict(x, return_std=True)
    if s == 0: s = 1e-9
    r = m - add * s
    return r


def run_bo(acquisition, max_iter, init=25, random=True, acq_add=1, seed=1):
    """
    BO
    :param max_iter: max number of function calls
    :param init: number of points to build initial model
    :param explore: factor to infuence exploration
    :param random: if False initial points are lsinearly sampled in the bounds, otherwise uniformly random.
    :return: all evaluated points
    """
    # sample initial query points
    np.random.seed(seed)
    if random:
        x = np.random.uniform(-15, 10, init).reshape(-1, 1).tolist()
    else:
        x = np.linspace(-15, 10, init).reshape(-1, 1).tolist()
    # get corresponding response values
    y = list(map(f, x))

    for i in range(max_iter - init):  # BO loop
        logging.debug('Sample #%d' % (init + i))
        #Feel free to adjust the hyperparameters
        gp = Pipeline([["standardize", StandardScaler()],
                      ["GP", GPR(kernel=Matern(nu=2.5), normalize_y=True, n_restarts_optimizer=10)], 
                    ])
        gp.fit(x, y)  # fit the model

        # Partially initialize the acquisition function to work with the fmin interface
        # (only the x parameter is not specified)
        # TODO implement different acquisition functions
        acqui = partial(acquisition, model=gp, eta=min(y), add=acq_add)
        # optimize acquisition function, repeat 10 times, use best result
        x_ = None
        y_ = 10000
        # Feel free to adjust the hyperparameters
        for i in range(10):
            opt_res = minimize(acqui, np.random.uniform(-15, 10), bounds=[[-15, 10]], options={"maxfun": 10}, method="L-BFGS-B")
            if opt_res.fun[0] < y_:
                x_ = opt_res.x
                y_ = opt_res.fun[0]
        x.append(x_)
        y.append(f(x_))
    return y

def run_rand(max_iter, seed):
    rng = np.random.RandomState(seed)
    x = rng.uniform(-15, 10, max_iter).reshape(-1, 1)
    return [f([i, ]) for i in x]

def main(num_evals, init_size, repetitions, random, seed):
    # Do some plots
    rng = np.random.RandomState(5)
    x = rng.uniform(-15, 10, 10).reshape(-1, 1)
    y = [f([i, ]) for i in x]
    gp = Pipeline([["standardize", StandardScaler()],
                  ["GP", GPR(kernel=Matern(nu=2.5), normalize_y=True, n_restarts_optimizer=10)], 
                ])
    gp.fit(x, y)  # fit the model
    

    x_axis = np.linspace(-15, 10, 500)
    y_func = [f([i, ]) for i in x_axis]
    
    ei = np.array([EI(i, gp, min(y)) for i in x_axis]).flatten()
    ucb = np.array([UCB(i, gp, min(y), 1) for i in x_axis]).flatten()
    m, s = gp.predict(x_axis.reshape([-1, 1]), return_std=True)
    m = m.flatten()
    s = s.flatten()
    plt.scatter(x, y)
    plt.plot(x_axis, y_func, label="true")
    plt.plot(x_axis, ei, label="ei")
    plt.plot(x_axis, ucb, label="ucb")
    plt.plot(x_axis, m, label="model")
    plt.fill_between(x_axis, m-s, m+s, alpha=0.5)
    #plt.yscale("log")
    plt.legend()
    plt.show()
    
    
    #sys.exit(1)
    # Actually run BO
    EI_res = []
    UCB_res = []
    random_res = []
    grid_res = []
    for i in range(repetitions):
        bo_res_1 = run_bo(max_iter=num_evals, init=init_size, random=random, acquisition=EI, acq_add=1, seed=seed+i)
        bo_res_2 = run_bo(max_iter=num_evals, init=init_size, random=random, acquisition=UCB, acq_add=1, seed=seed+i)
        rand_res = run_rand(max_iter=num_evals, seed=seed+i)
        # TODO implement random search
        # TODO implement grid search
        # TODO evaluation
        EI_res.append(np.minimum.accumulate(bo_res_1).flatten())
        UCB_res.append(np.minimum.accumulate(bo_res_2).flatten())
        random_res.append(np.minimum.accumulate(rand_res).flatten())

    for name, res in (["EI", EI_res], ["UCB", UCB_res], ["rand", random_res]):
        r = np.vstack(res)
        m = np.mean(res, axis=0)
        s = np.std(r, axis=0)
        plt.plot(m, label=name)
        plt.fill_between(np.arange(0, m.shape[0]), m+s, m-s, alpha=0.2)

    plt.yscale("log")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    cmdline_parser = argparse.ArgumentParser('ex06')

    cmdline_parser.add_argument('-n', '--num_func_evals',
                                default=100,
                                help='Number of function evaluations',
                                type=int)
    cmdline_parser.add_argument('-p', '--percentage_init',
                                default=0.25,
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
    cmdline_parser.add_argument('--seed',
                                default=0,
                                help='Which seed to use',
                                required=False,
                                type=int)
    cmdline_parser.add_argument('--repetitions',
                                default=5,
                                help='How often to repeat the experiment',
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
    main(num_evals=args.num_func_evals, init_size=init_size, repetitions=args.repetitions, random=args.random_initial_design, seed=args.seed)

import numpy as np
from scipy.stats import norm


def EI(x, model, eta, add=None, plotting=False):
    """
    Expected Improvement.
    :param x: point to determine the acquisition value
    :param model: GP to predict target function value
    :param eta: best so far seen value
    :param add: additional parameters necessary for the function (none)
    :param plotting: flag to fulfill fmin interface / show plots with functions to be maximized.
    :return: positive EI value for plotting, negative for the optimizer.
    """
    x = np.array([x]).reshape([-1, 1])
    mu, sigma = model.predict(x, return_std=True)

    with np.errstate(divide='warn'):
        improvement = mu - eta
        Z = improvement/sigma
        ei = improvement * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0
    # return negative value for the Python Scipy Optimize .minimize, positive for plots
    return (ei if plotting else -ei)

def UCB(x, model, eta, add, plotting=False):
    """
    Upper Confidence Bound
    :param x: point to determine the acquisition value
    :param model: GP to predict target function value
    :param eta: best so far seen value
    :param add: additional parameters necessary for the function (kappa)
    :param plotting: flag to fulfill fmin interface / show plots with functions to be maximized.
    :return: positive UCB value for plotting, negative for the optimizer.
    """
    x = np.array([x]).reshape([-1, 1])
    mu, sigma = model.predict(x, return_std=True)

    kappa = np.sqrt(add)
    ucb = mu + kappa * sigma
    return (ucb if plotting else -ucb)

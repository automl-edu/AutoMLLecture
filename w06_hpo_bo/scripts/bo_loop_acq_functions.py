import numpy as np
from scipy.stats import norm


def PI(x, model, eta, add, plotting=False):
    """
	Probability of Improvement
	:param x: point to determine the acquisition value
    :param model: GP to predict target function value
    :param eta: best so far seen value
    :param add: (epsilon) trade-off parameter (>=0)
    :param plotting: flag to fulfill fmin interface / show plots with functions to be maximized.
    :return: positive PI value for plotting, negative for the optimizer.
    """
    x = np.array([x]).reshape([-1, 1])
    mu, sigma = model.predict(x, return_std=True)
    Z = (mu - eta - add)/sigma
    pi = norm.cdf(Z)
    # return pi
    return -pi if plotting else pi



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
    # return ei
    return -ei if plotting else ei

def LCB(x, model, eta, add):
    """
    Lower Confidence Bound, returns a value for the minimizer
    :param x: point to determine the acquisition value
    :param model: GP to predict target function value
    :param eta: best so far seen value
    :param add: additional parameters necessary for the function (kappa)
    :return: positive LCB value for plotting, negative for the optimizer.
    """
    x = np.array([x]).reshape([-1, 1])
    mu, sigma = model.predict(x, return_std=True)

    lcb = mu - add * sigma
    return lcb
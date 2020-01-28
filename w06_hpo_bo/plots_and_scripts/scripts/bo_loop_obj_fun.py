import numpy as np

# following one of the AutoML assignments
# maximizing, so it's consistent with HPO problem
def f(x):
    """
    Function to maximize. (negative Levy1D: see https://www.sfu.ca/~ssurjano/levy.html & flip it). Global max value: 0.0
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
    return -y

# bounds for the search
bounds = dict({
    'lower': -15,
    'upper': 10
})

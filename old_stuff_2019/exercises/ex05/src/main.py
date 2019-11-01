import numpy as np

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


class DEOptimizer:
    """
    DE Optimizer
    :param max_iter: max number of iterations
    :param D: dimension of the problem being solved
    :param f: function to be optimized
    :param NP: population size
    :param F: scaling factor
    :param CR: crossover rate
    :return: the best individual with the best function value
    """

    def __init__(self):

        # TODO Initialize DE parameters and random population
        self.pop = []

    # TODO implement mutation operation
    def mutation(self):
        v = []
        return v

    # TODO implement crossover operation
    def crossover(self):
        u = []
        return u

    # TODO implement DE optimization loop
    def DE_optimization_loop(self):
        # start DE evolutionary search for max_iter iterations, and perform mutation, crossover and selection for each iteration

        # return the best individual x for each iteration
        x_best = []

        return x_best

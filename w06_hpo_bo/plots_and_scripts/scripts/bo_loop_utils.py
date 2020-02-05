from matplotlib import pyplot as plt
import numpy as np

from bo_loop_acq_functions import UCB, EI
from bo_loop_obj_fun import f, bounds


# Dictionaries for the graphs' appearance
colors = dict({
    'observations': 'green',
    'new_observation': 'black',
    'gp_mean': 'red',
    'gp_variance': 'red',
    'acq_fun': 'black'
})

labels = dict({
    UCB: 'Upper Confidence Bound',
    EI: 'Expected Improvement'
})

ylabels = dict({
    UCB: 'UCB(x)',
    EI: 'EI(x)'
})

acquisition_functions = dict({
    UCB: UCB,
    EI: EI,
    'UCB': UCB,
    'EI': EI
})

# Plot objective function, defined f(x)
def plot_objective_function():
    axis = np.arange(start=bounds['lower'], stop=bounds['upper'], step=0.1)
    plt.plot(axis, f([axis]), linestyle='--', label="Objective function")
    plt.legend()
    plt.grid()

# Plot objective function, observations (mark the newest) and the surrogate model (with mean and variance of GP)
def plot_search_graph(observed_x, observed_y, model):
    plt.figure(1)

    new_x = np.linspace(bounds['lower'], bounds['upper'], 100)
    mu, sigma = model.predict(new_x.reshape(-1, 1), return_std=True)

    plot_objective_function()
    plt.scatter(observed_x[:-1], observed_y[:-1], color=colors['observations'], label="Observations (" + str(len(observed_x)-1) + ")")
    plt.scatter(observed_x[-1], observed_y[-1], color=colors['new_observation'], marker='v', label="Newest observation")
    plt.plot(new_x, mu, lw=2, color=colors['gp_mean'], label="GP mean")
    plt.fill_between(new_x, mu+sigma, mu-sigma, facecolor=colors['gp_variance'], alpha=0.5, label="GP std")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title('Search graph')
    plt.legend()
    plt.show()

# Plot acquisition function
def plot_acquisition_function(acquisition, eta, model, add=None):
    plt.figure(2)

    new_x = np.linspace(bounds['lower'], bounds['upper'], 100)
    acquisition_fun = acquisition_functions[acquisition](new_x, model=model, eta=eta, add=add, plotting=True)
    zipped = list(zip(new_x, acquisition_fun))
    zipped.sort(key = lambda t: t[0])
    new_x, acquisition_fun = list(zip(*zipped))

    plt.plot(new_x, acquisition_fun, color=colors['acq_fun'], label=labels[acquisition])
    plt.xlabel("x")
    plt.ylabel(ylabels[acquisition])
    plt.title(labels[acquisition])
    plt.legend()
    plt.grid()
    plt.show()

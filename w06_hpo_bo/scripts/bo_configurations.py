import numpy as np
from bo_loop_acq_functions import *


# Dictionaries for the graphs' appearance
colors = dict({
    'observations': 'black',
    'highlighted_observations': 'green',
    'new_observation': 'red',
    'current_incumbent': 'red',
    'highlighted_point': 'red',
    'gp_mean': '#0F028A',
    'gp_variance': 'lightblue',
    'gp_variance_edge': 'k',
    'acq_fun': 'black',
    'envelope_min_opacity': 0.3,
    'envelope_max_opacity': 0.8,
    'minor_tick_highlight': 'red',
    'acq_func_fill': 'lightblue',
    'acq_func_intro': 'seagreen',
    'acq_func_intro_fill': 'mediumaquamarine'
})

# Various parameters for plotting required by our own code
params = {
    'sample_precision': 100, # Number of points to sample while plotting in the unit open interval [0, 1)
}

labels = dict({
    PI: 'Probability of Improvement',
    LCB: 'Lower Confidence Bound',
    EI: 'Expected Improvement',
    'xlabel': '$\lambda$',
    'gp_ylabel': 'c($\lambda$)',
    'acq_ylabel': "$u^{(t)}(\lambda')$",
    'gp_mean': 'GP Mean',
    'incumbent': 'Current Incumbent'
})

ylabels = dict({
    PI: 'PI(x)',
    LCB: 'LCB(x)',
    EI: 'EI(x)'
})


acquisition_functions = dict({
    PI: PI,
    LCB: LCB,
    EI: EI,
    'PI': PI,
    'LCB': LCB,
    'EI': EI
})


def f(x):
    return x[0]/5 * np.sin(x[0]) + x[0]/7 * np.cos(2 * x[0])

# bounds for the search
bounds = {
    "x": (2, 9),
    "gp_y": (-3, 3),
    "acq_y": (0, 5),
    "x_intro": (2, 13),
    "y_intro": (0, 15)

}

zorders = {
    'annotations_low': 20,
    'zone_of_imp': 30,
    'annotations_normal': 40,
    'datapoints': 41,
    'incumbent': 42,
    'annotations_high': 60,
    'legend': 100
}
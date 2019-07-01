import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from nas_cifar10 import NASCifar10A
from optimizers import RandomSearch as RS
from optimizers import Evolution

parser = argparse.ArgumentParser()
parser.add_argument('--run_id', default=0, type=int, nargs='?', help='unique number to identify this run')
parser.add_argument('--n_iters', default=100, type=int, nargs='?', help='number of iterations for optimization method')
parser.add_argument('--data_dir', default="./benchmark", type=str, nargs='?', help='specifies the path to the tabular data')
parser.add_argument('--pop_size', default=100, type=int, nargs='?', help='population size')
parser.add_argument('--sample_size', default=10, type=int, nargs='?', help='sample_size')
parser.add_argument('--n_repetitions', default=20, type=int, nargs='?', help='number of optimization independent runs')
args = parser.parse_args()

# load the tabular benchmark
b = NASCifar10A(data_dir=args.data_dir)

# collect the results from each optimization run here
rs_runs = []
re_runs = []
non_re_runs = []

for i in range(args.n_repetitions):
	# run random search for n_iters function evaluations on benchmark b
	rs = RS(b)
	rs.optimize(args.n_iters)
	rs_runs.append(rs.incumbent_trajectory_error)

	# run regularized evolution for n_iters function evaluations on benchmark b
	re = Evolution(b)
	re.optimize(args.n_iters, args.pop_size, args.sample_size)
	re_runs.append(re.incumbent_trajectory_error)

	# run non-regularized evolution for n_iters function evaluations on benchmark b
	non_re = Evolution(b)
	non_re.optimize(args.n_iters, args.pop_size, args.sample_size, False)
	non_re_runs.append(non_re.incumbent_trajectory_error)

# plotting the incumbents error
fig, ax = plt.subplots()
x = range(args.n_iters)
labels = ['RS', 'RE', 'non-RE']
c = ['b', 'r', 'g']
for i, setting in enumerate([rs_runs, re_runs, non_re_runs]):
	mean = np.mean(setting, axis=0)
	std = np.std(setting, axis=0)

	ax.plot(x, mean, c=c[i], label=labels[i])
	ax.fill_between(x, mean-std, mean+std, facecolor=c[i], alpha=0.3)
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_ylim([5e-2, 7e-1])

plt.legend()
plt.tight_layout()
plt.savefig('logs/plot.png')
plt.show()


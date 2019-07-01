import os
import argparse
import matplotlib.pyplot as plt

from nas_cifar10 import NASCifar10A
from optimizers import RandomSearch as RS
from optimizers import Evolution

parser = argparse.ArgumentParser()
parser.add_argument('--run_id', default=0, type=int, nargs='?', help='unique number to identify this run')
parser.add_argument('--n_iters', default=100, type=int, nargs='?', help='number of iterations for optimization method')
parser.add_argument('--data_dir', default="./benchmark", type=str, nargs='?', help='specifies the path to the tabular data')
parser.add_argument('--pop_size', default=100, type=int, nargs='?', help='population size')
parser.add_argument('--sample_size', default=10, type=int, nargs='?', help='sample_size')
args = parser.parse_args()

# load the tabular benchmark
b = NASCifar10A(data_dir=args.data_dir)

# run random search for n_iters function evaluations on benchmark b
rs = RS(b)
rs.optimize(args.n_iters)

# run regularized evolution for n_iters function evaluations on benchmark b
re = Evolution(b)
re.optimize(args.n_iters, args.pop_size, args.sample_size)

# run non-regularized evolution for n_iters function evaluations on benchmark b
non_re = Evolution(b)
non_re.optimize(args.n_iters, args.pop_size, args.sample_size, False)

plt.plot(range(args.n_iters), rs.incumbent_trajectory_error, c='r', label='RS')
plt.plot(range(args.n_iters), re.incumbent_trajectory_error, c='b', label='RE')
plt.plot(range(args.n_iters), non_re.incumbent_trajectory_error, c='g', label='non-RE')
plt.yscale('log')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('plot.png')
plt.show()



import numpy as np
import collections
import os
import random
import ConfigSpace as CS

from copy import deepcopy
from nas_cifar10 import NASCifar10A


class NASOptimizer(object):
    """
    Base class for NASBench-101 optimizers. All subclasses should
    inherit from this.
    """
    def __init__(self, benchmark: NASCifar10A):
        # get the configuration space
        self.benchmark = benchmark
        self.cs = benchmark.get_configuration_space()

        # incumbent_trajectory keeps track of the best
        # configuration (architecture) at each point in time.
        # incumbent_trajectory_error keeps track of the
        # corresponding validation errors of incumbent_trajectory
        self.incumbent_trajectory = []
        self.incumbent_trajectory_error = []
        self.curr_wallclock = 0
        self.curr_incumbent = None
        self.curr_incumbent_error = np.inf

    def optimize(self, n_iters: int = 100):
        raise NotImplementedError

    def sample_random_config(self):
        """
        Return a randomly sampled configuration.
        """
	# TODO: return one randomly sampled configuration from self.cs
        
	return config

    def train_and_eval(self, config: CS.Configuration):
        """
        Function that queries the validation error of config
        in self.benchmark. Since every architecture has 
        already been trained and evaluated, we just do table
        look-ups without the need to train the neural net.
        """
        # TODO: query validation error from the tabular benchmark

        # TODO: check if config is better than current incumbent
	
	# TODO: updated the incumbent trajectory

	# return the validation error and cost of the queried config
        return y, cost


class RandomSearch(NASOptimizer):
    """
    Algorithm for random search.
    """
    def __init__(self, benchmark: NASCifar10A):
        super(RandomSearch, self).__init__(benchmark)

    def optimize(self, n_iters: int = 100):
        """
        Run random search for n_iters function evaluations.
        """
        for i in range(n_iters):
            config = self.sample_random_config()
            self.train_and_eval(config)


class Evolution(NASOptimizer):
    """Algorithm for regularized evolution (i.e. aging evolution).

    Follows "Algorithm 1" in Real et al. "Regularized Evolution for Image
    Classifier Architecture Search".

    """
    def __init__(self, benchmark: NASCifar10A):
        super(Evolution, self).__init__(benchmark)
        
        self.population = collections.deque()
        self.history = {}  # keep track of all sampled configurations

    def optimize(self, n_iters: int = 100, population_size: int = 100,
                 sample_size: int = 20, regularized: bool = True):
        """
        Args:
          n_iters: the number of iterations the algorithm should run for.
          population_size: the number of individuals to keep in the population.
          sample_size: the number of individuals that should participate in each
              tournament.
        """
        # Initialize the population with random models.
        while len(self.population) < population_size:
            config = self.sample_random_config()
            y, _ = self.train_and_eval(config)

            self.population.append(config)
            self.history[config] = y

        # Carry out evolution in cycles (n_iters). Each cycle produces a model and removes
        # another.
        while len(self.history) < n_iters:
            # Sample randomly chosen models from the current population.
            sample = {} # keys: config, values: validation errors
            while len(sample) < sample_size:
	    	# TODO: randomly choose one architecture from population
		#       and add it to sample

            # The parent is the best model in the sample.
	    # TODO: get the best model in the current sample
            parent = None

            # TODO: Create the child model and evaluate it.
            
            # TODO: add the child to the population and history

	    # TODO: based on the regularized argument value
	    #       remove the oldest or the worst architecture
	    #       from the population


    def mutate_arch(self, parent_arch):
        """
        Apply mutation to a parent architecture and return the mutated one.
        """
        # pick random parameter
        dim = np.random.randint(len(self.cs.get_hyperparameters()))
        hyper = self.cs.get_hyperparameters()[dim]

        if type(hyper) == CS.OrdinalHyperparameter:
            choices = list(hyper.sequence)
        else:
            choices = list(hyper.choices)
        # drop current values from potential choices
        choices.remove(parent_arch[hyper.name])

        # flip parameter
        idx = np.random.randint(len(choices))

        child_arch = deepcopy(parent_arch)
        child_arch[hyper.name] = choices[idx]
        return child_arch
        

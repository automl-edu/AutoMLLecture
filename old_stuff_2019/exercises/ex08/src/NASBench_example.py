import random
from nas_cifar10 import NASCifar10A

random.seed(43)

# Load the benchmark
b = NASCifar10A(data_dir="./benchmark")
cs = b.get_configuration_space()

# Sample one random configuration/architecture
config = cs.sample_configuration()

# Query the validation error and runtime 
# cost from the tabular benchmark for config
y, cost = b.objective_function(config)

print("Numpy representation: ", config.get_array())
print("Dict representation: ", config.get_dictionary())
print("Validaton error: ", y)
print("Runtime: ", cost)


import argparse
import logging

import numpy as np

from smac.optimizer.objective import average_cost, _cost, _runtime
from smac.runhistory.runhistory import RunHistory
from smac.scenario.scenario import Scenario


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Parse CMDline arguments and basic setup
parser = argparse.ArgumentParser('Evaluate SMAC results')
parser.add_argument('data', help='File in json format that contains all validated runs')
parser.add_argument('scenario', help='Scenario file')
args, unkown = parser.parse_known_args()

logging.basicConfig(level=logging.INFO)
if unkown:
    logging.warning('Could not parse the following arguments: ')
    logging.warning(str(unkown))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Create Runhistory object as well as scenario object
runhist = RunHistory(average_cost)
scenario = Scenario(args.scenario, cmd_args={'output_dir': ""})
cs = scenario.cs

runhist.load_json(args.data, cs)  # populate the runhistory with the validation data
configs = runhist.get_all_configs()
def_ = cs.get_default_configuration()
def_dict = def_.get_dictionary()

# Switch it around such that statistics about the default are gathered first
if configs[0] != def_:
    tmp = configs[0]
    configs[0] = configs[1]
    configs[1] = tmp
    del tmp
logging.info('Found %d configs' % len(configs))
logging.info('Cost per config:')

# For each config
for config in configs:
    # gather statistics such as
    config_dict = config.get_dictionary()
    costs = np.array(_cost(config, runhist))    # the cost for running on each instance
    runtime = np.mean(np.array(_runtime(config, runhist)))  # the mean runtime
    cost = np.mean(costs)  # the mean cost
    timeouts = np.sum(costs > scenario.cutoff)  # and count the number of timeouts
    default = config == def_

    # This is just cosmetics for the output
    config_str = config.__repr__().split('\n')  # get the string representation
    num_changed = 0
    for idx, line in enumerate(config_str):
        try:
            name, value = list(map(lambda x: x.strip(), line.split(', ')))
            if name in def_dict:
                changed = config_dict[name] != def_dict[name]
            else:
                changed = True
            num_changed += changed
            v_str, val = value.split(': ')
            config_str[idx] = '{:>30}: {:>23} |    {:1s}    |'.format(name, val, 'X' if changed else ' ')
        except ValueError:
            if not line:
                del config_str[idx]
            pass

    if not default:  # count the number of parameters that are unset
        d_keys = set(def_dict.keys())
        c_keys = set(config_dict.keys())
        l_diff = set.difference(d_keys, c_keys)
        # r_diff = set.difference(c_keys, d_keys)  # No need to count the right diff as it will automatically be counted above
        diff = set.union(l_diff, set())
        num_changed += len(diff)
        for item in l_diff:
            config_str.append('{:>30}: {:>23} |    {:1s}    |'.format(item, 'None', 'X'))
    config_str[0] = '{:>{width}s}{:>{filler}s} | {:^7s} | {:>9s}  {:> 8.5f}'.format(config_str[0], ' ',
                                                                                   'Changed' if not default else ' ',
                                                                                   'Cost:', cost,
                                                                                   width=len(config_str[0]),
                                                                                   filler=55 - len(config_str[0]))
    config_str[1] += ' {:>9s}  {:> 8.5f}'.format('Runtime:', runtime)
    config_str[2] += ' {:>9s} {:> 3d}'.format('Timeouts:', timeouts)
    if not default:
        config_str[3] += ' {:>9s} {:>3d}'.format('Changed:', num_changed)
    else:
        config_str[3] += ' {:>8s}'.format('Default')
    print('\n'.join(config_str))
    print()

import argparse
import logging

import numpy as np
import matplotlib.pyplot as plt


def load_data(fl="data.csv"):
    """
    Loads data stored in fl
    :param fl: filename of csv file
    :return: y1, y2
    """
    data = np.loadtxt(fl, delimiter=",")
    y1 = data[:, 0]
    y2 = data[:, 1]
    return y1, y2


def paired_permutation_test(data_A, data_B, repetitions=10000) -> float:
    """
    TODO
    :param data_A: runs of configuration a
    :param data_B: runs of configuration b
    :param repetitions: number of repetitions to use for the test
    :return:p-value
    """
    p_value = np.random.uniform(0, 1)
    return p_value


def cdf_plot(data_A, data_B):
    """
    TODO
    :param data_A: runs of configuration a
    :param data_B: runs of configuration b  
    """
    pass

def scatter_plot(data_A, data_B):
    """
    TODO
    :param data_A: runs of configuration a
    :param data_B: runs of configuration b  
    """

    pass

def box_plot(data_A, data_B):
    """
    TODO
    :param data_A: runs of configuration a
    :param data_B: runs of configuration b  
    """
    pass

def violin_plot(data_A, data_B):
    """
    TODO
    :param data_A: runs of configuration a
    :param data_B: runs of configuration b  
    """
    pass


def main(args):
    logging.info('Loading data')
    data_A, data_B = load_data(fl="./data.csv")
    
    # (a)
    # TODO
    scatter_plot(data_A, data_B)

    # (b)
    # TODO
    cdf_plot(data_A, data_B)

    # (c)
    # TODO
    box_plot(data_A, data_B)
    violin_plot(data_A, data_B)

    # (d)
    # TODO
    alpha = None
    statistic = paired_permutation_test(data_A, data_B, repetitions=10000)

if __name__ == '__main__':
    cmdline_parser = argparse.ArgumentParser('ex03')

    cmdline_parser.add_argument('-v', '--verbose', default='INFO', choices=['INFO', 'DEBUG'], help='verbosity')
    cmdline_parser.add_argument('--seed', default=12345, help='Which seed to use', required=False, type=int)
    args, unknowns = cmdline_parser.parse_known_args()
    np.random.seed(args.seed)
    log_lvl = logging.INFO if args.verbose == 'INFO' else logging.DEBUG
    logging.basicConfig(level=log_lvl)

    if unknowns:
        logging.warning('Found unknown arguments!')
        logging.warning(str(unknowns))
        logging.warning('These will be ignored')
    main(args)

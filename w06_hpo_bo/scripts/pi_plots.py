import warnings
warnings.filterwarnings('ignore')
import argparse
import logging
import os.path
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import Matern

from matplotlib import pyplot as plt

import bo_plot_utils as boplot
from bo_configurations import *


SEED = None
TOGGLE_PRINT = False
INIT_X_PRESENTATION = [2.5, 3.5, 5.5, 7, 9]
OUTPUT_DIR = os.path.abspath("./outputs/pi")
bounds["x"] = (2, 13)
bounds["gp_y"] = (-5, 5)

boplot.set_rc("savefig", directory=OUTPUT_DIR)

labels["xlabel"] = "$\lambda$"
labels["gp_ylabel"] = ""


def initialize_dataset(initial_design, init=None):
    """
    Initialize some data to start fitting the GP on.
    :param initial_design: Method for initializing the GP, choice between 'uniform', 'random', and 'presentation'
    :param init: Number of datapoints to initialize with, if relevant
    :return:
    """

    x = None

    # sample initial query points
    if initial_design == 'uniform':
        x = np.linspace(bounds["x"][0], bounds["x"][1], init).reshape(-1, 1).tolist()
    elif initial_design == 'random':
        x = np.random.uniform(bounds["x"][0], bounds["x"][1], init).reshape(-1, 1).tolist()
    elif initial_design == 'presentation':
        x = np.array(INIT_X_PRESENTATION).reshape(-1, 1).tolist()

    # get corresponding response values
    y = list(map(f, x))

    return x, y


def get_mu_star(model):
    """
    Given a model, return the (x, y) coords of mu-star.
    :param model: The underlying GP.
    :return:
    """

    X_ = boplot.get_plot_domain()
    mu = model.predict(X_).reshape(-1, 1)

    coords = np.hstack((X_, mu))
    idx = np.argmin(coords, axis=0)[1]
    return coords[idx, :]


def visualize_pi(initial_design, init=None):
    """
    Visualize one-step of PI.
    :param initial_design: Method for initializing the GP, choice between 'uniform', 'random', and 'presentation'
    :param init: Number of datapoints to initialize GP with.
    :return: None
    """

    # 1. Plot GP fit on initial dataset
    # 2. Mark current incumbent
    # 3. Mark Zone of Probable Improvement
    # 4. Draw Vertical Normal at a good candidate for improvement
    # 5. Draw Vertical Normal at a bad candidate for improvement


    logging.debug("Visualizing PI with initial design {} and init {}".format(initial_design, init))
    # Initialize dummy dataset
    x, y = initialize_dataset(initial_design=initial_design, init=init)
    ymin_arg = np.argmin(y)
    ymin = y[ymin_arg]
    logging.debug("Initialized dataset with:\nsamples {0}\nObservations {1}".format(x, y))

    # Fit GP to the currently available dataset
    gp = GPR(kernel=Matern())
    logging.debug("Fitting GP to\nx: {}\ny:{}".format(x, y))
    gp.fit(x, y)  # fit the model
    mu_star_t_xy = get_mu_star(gp)
    logging.info("Mu-star at time t: {}".format(mu_star_t_xy))

    # noinspection PyStringFormat
    logging.debug("Model fit to dataset.\nOriginal Inputs: {0}\nOriginal Observations: {1}\n"
                  "Predicted Means: {2}\nPredicted STDs: {3}".format(x, y, *(gp.predict(x, return_std=True))))

    def draw_basic_plot(mark_incumbent=True, show_objective=False):
        fig, ax = plt.subplots(1, 1, squeeze=True)
        ax.set_xlim(bounds["x"])
        ax.set_ylim(bounds["gp_y"])
        ax.grid()
        boplot.plot_gp(model=gp, confidence_intervals=[1.0, 2.0, 3.0], custom_x=x, ax=ax)
        if show_objective:
            boplot.plot_objective_function(ax=ax)
        boplot.mark_observations(X_=x, Y_=y, mark_incumbent=mark_incumbent,
                                 highlight_datapoint=None, highlight_label=None, ax=ax)

        if mark_incumbent:
            boplot.highlight_output(
                y=np.array([ymin]),
                label=['$c_{inc}$'],
                lloc='left',
                ax=ax,
                disable_ticks=True
            )
            boplot.annotate_y_edge(
                label='$c_{inc}$',
                xy=(x[ymin_arg], ymin),
                ax=ax,
                align='left',
                yoffset=0.8
            )

        return fig, ax


    def finishing_touches(ax, show_legend=True, figname='pi.pdf'):
        ax.set_xlabel(labels['xlabel'])
        # ax.set_ylabel(labels['gp_ylabel'])
        # ax.set_title(r"Visualization of $\mathcal{G}^{(t)}$", loc='left')

        if show_legend:
            ax.legend().set_zorder(zorders['legend'])
        else:
            ax.legend().remove()

        plt.tight_layout()
        if TOGGLE_PRINT:
            plt.savefig(f"{OUTPUT_DIR}/{figname}")
        else:
            plt.show()

    # 1. Plot GP fit on initial dataset
    # -------------Plotting code -----------------

    fig, ax = draw_basic_plot(mark_incumbent=False, show_objective=True)

    finishing_touches(ax, show_legend=True, figname="pi_1.pdf")

    # -------------------------------------------
    # 2. Mark current incumbent
    # -------------Plotting code -----------------

    fig, ax = draw_basic_plot(mark_incumbent=True, show_objective=True)

    finishing_touches(ax, show_legend=True, figname="pi_2.pdf")

    # -------------------------------------------

    def draw_final_graph(show_objective=False, show_vertical_normals=True, candidates=None, normal_labels=None):
        if candidates is None and show_vertical_normals:
            raise RuntimeError("In order to show vertical normal distributions, candidates at which the PDF is sampled "
                               "must be specified as a list of floats.")

        fig, ax = draw_basic_plot(show_objective=show_objective)
        boplot.darken_graph(y=ymin, ax=ax)

        if show_vertical_normals:
            if normal_labels is None:
                normal_labels = [r"$P(\lambda_{%d})$" % (i + 1) for i in range(len(candidates))]
            elif type(normal_labels) is str:
                normal_labels = [normal_labels] * len(candidates)

            for idx in range(len(candidates)):
                candidate = candidates[idx]
                label = normal_labels[idx]
                vcurve_x, vcurve_y, mu = boplot.draw_vertical_normal(
                    gp=gp, incumbenty=ymin, ax=ax, xtest=candidate,
                    xscale=2.0, yscale=1.0
                )

                # ann_x = candidate + 0.5 * (np.max(vcurve_x) - candidate) / 2
                ann_x = candidate
                # ann_y = ymin - 0.25
                ann_y = ymin - 3.0

                arrow_x = candidate + 0.5 * (np.max(vcurve_x) - candidate) / 2
                arrow_x = ann_x + 0.1
                # arrow_y = ann_y - 3.0
                arrow_y = ymin - 3.0

                # prob = "{:.2f}".format(candidate)

                ax.annotate(
                    s=label, xy=(ann_x, ann_y), xytext=(arrow_x, arrow_y),
                    # arrowprops={'arrowstyle': 'fancy', 'shrinkA': 20.0},
                    # weight='heavy', color='darkgreen', zorder=zorders['annotations_high']
                )
        return fig, ax


    # 3. Remove objective function.
    # -------------Plotting code -----------------

    fig, ax = draw_basic_plot(mark_incumbent=True, show_objective=False)

    finishing_touches(ax, show_legend=True, figname="pi_3.pdf")

    # -------------------------------------------

    # 4. Mark Zone of Probable Improvement (without legend and objective)
    # -------------Plotting code -----------------

    fig, ax = draw_final_graph(show_objective=False, show_vertical_normals=False)

    finishing_touches(ax, show_legend=False, figname="pi_4.pdf")

    # -------------------------------------------

    # 5. Draw Vertical Normal at a good candidate for improvement
    # -------------Plotting code -----------------
    fig, ax = draw_final_graph(
        show_vertical_normals=True,
        candidates=[5.0],
        normal_labels=r'$PI^{(t)} \approx 0.5$'
    )

    finishing_touches(ax, show_legend=False, figname="pi_5.pdf")
    # -------------------------------------------

    # 6. Draw Vertical Normal at a bad candidate for improvement
    # -------------Plotting code -----------------

    fig, ax = draw_final_graph(
        show_vertical_normals=True,
        candidates=[5.0, 8.0],
        normal_labels=[r'$PI^{(t)} \approx 0.5$', r'$PI^{(t)} \approx 0.0$']
    )

    finishing_touches(ax, show_legend=False, figname="pi_6.pdf")
    # -------------------------------------------


def main(init_size, initial_design):
        visualize_pi(
            init=init_size,
            initial_design=initial_design,
        )



if __name__ == '__main__':
    cmdline_parser = argparse.ArgumentParser('AutoMLLecture')

    cmdline_parser.add_argument('-f', '--init_db_size',
                                default=4,
                                help='Size of the initial database',
                                type=int)
    cmdline_parser.add_argument('-i', '--initial_design',
                                default="presentation",
                                choices=['random', 'uniform', 'presentation'],
                                help='How to choose first observations.')
    cmdline_parser.add_argument('-v', '--verbose',
                                default=False,
                                help='verbosity',
                                action='store_true')
    cmdline_parser.add_argument('-s', '--seed',
                                default=15,
                                help='Which seed to use',
                                required=False,
                                type=int)
    cmdline_parser.add_argument('-p', '--print',
                                default=False,
                                help='Print graphs to file instead of displaying on screen.',
                                action='store_true')

    args, unknowns = cmdline_parser.parse_known_args()
    log_lvl = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_lvl)

    if unknowns:
        logging.warning('Found unknown arguments!')
        logging.warning(str(unknowns))
        logging.warning('These will be ignored')

    # init_size = max(1, int(args.num_func_evals * args.fraction_init))
    # Seed the RNG to obtain reproducible results
    SEED = args.seed
    np.random.seed(SEED)

    TOGGLE_PRINT = args.print
    if TOGGLE_PRINT:
        boplot.enable_printing()
    else:
        boplot.enable_onscreen_display()

    #init_size = max(1, int(args.num_func_evals * args.fraction_init))

    main(
        init_size=args.init_db_size,
        initial_design=args.initial_design,
    )
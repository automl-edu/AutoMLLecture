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
OUTPUT_DIR = os.path.abspath("./outputs/ei")

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


def visualize_ei(initial_design, init=None):
    """
    Visualize one-step of EI.
    :param initial_design: Method for initializing the GP, choice between 'uniform', 'random', and 'presentation'
    :param init: Number of datapoints to initialize GP with.
    :return: None
    """

    logging.debug("Visualizing EI with initial design {} and init {}".format(initial_design, init))
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

    # --------------------------------------------

    def draw_basic_figure(obj_func=False):
        fig, ax = plt.subplots(1, 1, squeeze=True)
        ax.set_xlim(bounds["x"])
        ax.set_ylim(bounds["gp_y"])
        ax.grid()
        boplot.plot_gp(model=gp, confidence_intervals=[1.0, 2.0, 3.0], custom_x=x, ax=ax)
        if obj_func:
            boplot.plot_objective_function(ax=ax)
        boplot.mark_observations(X_=x, Y_=y, mark_incumbent=True, highlight_datapoint=None, highlight_label=None, ax=ax)
        boplot.highlight_output(
            y=np.array([ymin]),
            label=['$c_{inc}$'],
            lloc='left',
            ax=ax,
            # disable_ticks=True
        )
        # boplot.annotate_y_edge(
        #     label='$c_{inc}$',
        #     xy=((ax.get_xlim()[0] + x[ymin_arg]) / 2, ymin),
        #     ax=ax,
        #     align='left',
        #     yoffset=1.0
        # )

        return fig, ax


    def perform_finishing_tasks(ax, filename="", remove_legend=True):

        ax.legend().set_zorder(zorders['annotations_high'])
        ax.set_xlabel(labels['xlabel'])

        if remove_legend:
            ax.legend().remove()

        plt.tight_layout()
        if TOGGLE_PRINT:
            plt.savefig(f"{OUTPUT_DIR}/{filename}")
        else:
            plt.show()

    # 1. Plot GP fit on initial dataset
    # -------------Plotting code -----------------
    fig, ax = draw_basic_figure(obj_func=True)

    perform_finishing_tasks(
        ax=ax,
        filename="ei_1.pdf",
        remove_legend=False
    )
    # -------------------------------------------

    def draw_basic_figure_plus_zone():
        fig, ax = draw_basic_figure(obj_func=False)
        boplot.darken_graph(y=ymin, ax=ax)

        return fig, ax


    # 2a. Mark Zone of Probable Improvement + Display Legend
    # -------------Plotting code -----------------
    fig, ax = draw_basic_figure_plus_zone()

    perform_finishing_tasks(
        ax=ax,
        filename="ei_2a.pdf",
        remove_legend=False
    )
    # -------------------------------------------


    # 2b. Mark Zone of Probable Improvement + Remove Legend
    # -------------Plotting code -----------------
    fig, ax = draw_basic_figure_plus_zone()

    perform_finishing_tasks(
        ax=ax,
        filename="ei_2b.pdf",
        remove_legend=True
    )
    # -------------------------------------------

    def draw_distribution_for_candidate(ax, candidate, target_cost):
        vcurve_x, vcurve_y, mu = boplot.draw_vertical_normal(
            gp=gp, incumbenty=ymin, ax=ax, xtest=candidate,
            xscale=2.0, yscale=1.0, fill=False, draw_domain=False
        )

        idx = np.where(np.logical_and(vcurve_y > target_cost - 0.1, vcurve_y < target_cost + 0.1))
        ann_y = vcurve_y[idx]
        ann_x = vcurve_x[idx]
        ax.fill_betweenx(ann_y, candidate, ann_x, alpha=1.0, facecolor='darkgreen',
                         zorder=zorders['annotations_high'] - 5)

    def draw_final_figure(sample_cost, vis_confs, inc_eq_loc_x, draw_improvement=True, draw_normals=True):
        fig, ax = draw_basic_figure_plus_zone()

        labels = [r'$\lambda_%d$' % (idx + 1) for idx in range(len(vis_confs))]
        boplot.highlight_configuration(
            x=np.array(vis_confs),
            label=labels,
            lloc='bottom',
            ax=ax,
            disable_ticks=True
        )
        for label, conf in zip(labels, vis_confs):
            boplot.annotate_x_edge(
                label=label,
                xy=(conf + 0.6 * (ax.get_xlim()[1] - ax.get_xlim()[0]) / 10, ymin),
                ax=ax,
                align='bottom',
                offset_param=1.9
            )
        boplot.highlight_output(
            y=np.array([sample_cost, ymin]),
            label=['c', '$c_{inc}$'],
            lloc='left',
            ax=ax
        )
        # boplot.annotate_y_edge(label=r'c', xy=(lambda, cost), align='left', ax=ax)

        if draw_improvement:
            ax.annotate(s='', xy=(inc_eq_loc_x, sample_cost), xytext=(inc_eq_loc_x, ymin),
                        arrowprops={'arrowstyle': 'simple', })

            ax.text(inc_eq_loc_x - 1.0, sample_cost - 1.0, r'$I_c=c_{inc}-c$', weight='heavy')

        if draw_normals:
            for idx in range(len(vis_confs)):
                conf = vis_confs[idx]
                draw_distribution_for_candidate(ax=ax, candidate=conf, target_cost=sample_cost)

                ax.annotate(
                    s=r"$p(c|\lambda_%d)$" % (idx+1), xy=(conf, sample_cost), xytext=(conf-1.8, sample_cost - 1.5),
                    arrowprops={'arrowstyle': 'fancy', 'shrinkA': 20.0},
                    weight='heavy', color='darkgreen', zorder=zorders['annotations_high']
                )


        return fig, ax

    # 3. Mark Hypothetical Real cost of a random configuration
    # -------------------------------------------

    candidate1 = 4.5
    candidate2 = 11

    fig, ax = draw_final_figure(
        sample_cost=-1.5,
        vis_confs=[candidate1],
        inc_eq_loc_x=None,
        draw_improvement=False,
        draw_normals=False
    )

    perform_finishing_tasks(
        ax=ax,
        filename="ei_3.pdf",
        remove_legend=True
    )
    # -------------------------------------------

    # 4. Display I(lambda)
    # -------------------------------------------

    fig, ax = draw_final_figure(
        sample_cost=-1.5,
        vis_confs=[candidate1],
        inc_eq_loc_x=(candidate1 + candidate2) / 2,
        draw_improvement=True,
        draw_normals=False
    )

    perform_finishing_tasks(
        ax=ax,
        filename="ei_4.pdf",
        remove_legend=True
    )
    # -------------------------------------------

    # 5. Display Vertical Normal Distribution
    # -------------------------------------------

    fig, ax = draw_final_figure(
        sample_cost=-1.5,
        vis_confs=[candidate1],
        inc_eq_loc_x=(candidate1 + candidate2) / 2
    )

    perform_finishing_tasks(
        ax=ax,
        filename="ei_5.pdf",
        remove_legend=True
    )
    # -------------------------------------------

    # 6. Display improvement for c_1 with two configurations
    # -------------------------------------------

    fig, ax = draw_final_figure(
        sample_cost=-1.5,
        vis_confs=[candidate1, candidate2],
        inc_eq_loc_x=(candidate1 + candidate2) / 2
    )

    perform_finishing_tasks(
        ax=ax,
        filename="ei_6.pdf",
        remove_legend=True
    )
    # -------------------------------------------


    # 7. Display improvement for c_2 with two configurations
    # -------------------------------------------

    fig, ax = draw_final_figure(
        sample_cost=-2.5,
        vis_confs=[candidate1, candidate2],
        inc_eq_loc_x=(candidate1 + candidate2) / 2
    )

    perform_finishing_tasks(
        ax=ax,
        filename="ei_7.pdf",
        remove_legend=True
    )
    # -------------------------------------------


    # 8. Display improvement for c_3 with two configurations
    # -------------------------------------------

    fig, ax = draw_final_figure(
        sample_cost=-3.0,
        vis_confs=[candidate1, candidate2],
        inc_eq_loc_x=(candidate1 + candidate2) / 2
    )

    perform_finishing_tasks(
        ax=ax,
        filename="ei_8.pdf",
        remove_legend=True
    )
    # -------------------------------------------


def main(init_size, initial_design):
    visualize_ei(
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

    # init_size = max(1, int(args.num_func_evals * args.fraction_init))

    main(
        init_size=args.init_db_size,
        initial_design=args.initial_design,
    )
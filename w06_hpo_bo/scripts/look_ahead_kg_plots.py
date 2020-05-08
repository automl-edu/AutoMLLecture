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
INIT_X_PRESENTATION = [2.5, 4, 6, 7, 8]
OUTPUT_DIR = os.path.abspath("./outputs/lookahead_kg")

labels["xlabel"] = "$\lambda$"
# colors['highlighted_observations'] = 'red'
# labels["gp_ylabel"] = "$c(\lambda')$"

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


def visualize_look_ahead(initial_design, init=None):
    """
    Visualize one-step of look-ahead.
    :param initial_design: Method for initializing the GP, choice between 'uniform', 'random', and 'presentation'
    :param init: Number of datapoints to initialize GP with.
    :return: None
    """

    # boplot.set_rcparams(**{'legend.loc': 'lower left'})

    logging.debug("Visualizing Look-Ahead with initial design {} and init {}".format(initial_design, init))
    # Initialize dummy dataset
    x, y = initialize_dataset(initial_design=initial_design, init=init)
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

    # Assume next evaluation location
    x_ = np.array([[5.0]])
    print(x_)
    y_ = f(x_[0])

    # Update dataset with new observation
    X2_ = np.append(x, x_, axis=0)
    Y2_ = y + [y_]

    logging.info("x: {}, y: {}".format(x_, y_))

    # Fit GP to the updated dataset
    gp2 = GPR(kernel=Matern())
    logging.debug("Fitting GP to\nx: {}\ny:{}".format(X2_, Y2_))
    gp2.fit(X2_, Y2_)  # fit the model
    mu_star_t1_xy = get_mu_star(gp2)
    logging.info("Mu-star at time t+1: {}".format(mu_star_t1_xy))

    # -------------------------Plotting madness begins---------------------------

    def draw_basic_figure(tgp=gp, tx=x, tX_=x, tY_=y, title='', highlight_datapoint=None, highlight_label="", ax=None):
        if ax is None:
            fig, ax = plt.subplots(1, 1, squeeze=True)
            plt.subplots_adjust(0.05, 0.15, 0.95, 0.85)
            figflag = True
        else:
            figflag = False

        ax.set_xlim(bounds["x"])
        ax.set_ylim(bounds["gp_y"])
        if title:
            ax.set_title(title, loc='left')
        ax.grid()
        boplot.plot_objective_function(ax=ax)
        boplot.plot_gp(model=tgp, confidence_intervals=[1.0, 2.0, 3.0], ax=ax, custom_x=tx)
        if highlight_datapoint:
            boplot.mark_observations(X_=tX_, Y_=tY_, mark_incumbent=False, ax=ax,
                                     highlight_datapoint=highlight_datapoint, highlight_label=highlight_label)
        else:
            boplot.mark_observations(X_=tX_, Y_=tY_, mark_incumbent=False, ax=ax)

        if figflag:
            return fig, ax
        else:
            return ax


    def perform_finishing_tasks(ax, filename="", remove_legend=True):
        ax.legend().set_zorder(zorders['legend'])
        ax.set_xlabel(labels['xlabel'])

        if remove_legend:
            ax.legend().remove()

        # plt.tight_layout()
        if TOGGLE_PRINT:
            # plt.savefig(f"{OUTPUT_DIR}/{filename}", bbox_inches='tight')
            plt.savefig(f"{OUTPUT_DIR}/{filename}")
        else:
            plt.show()

    # ---------------------------------------
    # Draw look ahead 1.

    labels['gp_mean'] = r'Mean: $\mu^{(t)}(\cdot)$'

    fig, ax = draw_basic_figure(title="")
    perform_finishing_tasks(ax=ax, filename="look_ahead_1.pdf", remove_legend=False)

    # ---------------------------------------
    # Draw look ahead 2
    fig, ax = draw_basic_figure(title="")

    logging.debug("Placing vertical on configuration: {}".format(x_))
    # boplot.highlight_configuration(x=x_, label='', lloc='bottom', ax=ax, ha='center')
    boplot.highlight_configuration(x=x_, label='', lloc='bottom', ax=ax, disable_ticks=True)
    boplot.annotate_x_edge(label=r'$\lambda$', xy=(x_, y_), align='bottom', ax=ax)

    perform_finishing_tasks(ax=ax, filename="look_ahead_2.pdf", remove_legend=True)

    # ---------------------------------------
    # Draw look ahead 3

    fig, ax = draw_basic_figure(title="")

    boplot.highlight_configuration(x=x_, label='', lloc='bottom', ax=ax, ha='right')
    boplot.annotate_x_edge(label=r'$\lambda$', xy=(x_, y_), align='bottom', ax=ax)

    boplot.highlight_output(y_, label='', lloc='right', ax=ax, fontsize=28)
    boplot.annotate_y_edge(label=r'$c(\lambda)$', xy=(x_, y_), align='right', ax=ax)

    ax.scatter(
        x_,
        y_,
        color=colors['highlighted_observations'],
        marker='X',
        label=r"Hypothetical Observation $<\lambda, c(\lambda)>$",
        zorder=zorders['annotations_normal']
    )

    perform_finishing_tasks(ax=ax, filename="look_ahead_3.pdf", remove_legend=True)

    # ---------------------------------------
    # Draw look ahead 4.

    labels['gp_mean'] = r'Mean: $\mu^{(t+1)}(\cdot)|_\lambda$'

    fig, ax = draw_basic_figure(tgp=gp2, tx=x, tX_=X2_, tY_=Y2_, title='',
                                highlight_datapoint=np.where(np.isclose(X2_, x_))[0],
                                highlight_label=r"Hypothetical Observation $<\lambda, c(\lambda)>$")

    perform_finishing_tasks(ax=ax, filename="look_ahead_4.pdf", remove_legend=False)

    # ---------------------------------------
    # Vertical comparison of look-ahead at any given x

    def draw_vertical_comparison(imaginary_lambda, ax1, ax2):
        tx_ = np.array([[imaginary_lambda]])
        ty_ = f(tx_[0])

        # Update dataset with new observation
        tX_ = np.append(x, tx_, axis=0)
        tY_ = y + [ty_]

        logging.info("x: {}, y: {}".format(tx_, ty_))

        # Fit GP to the updated dataset
        tgp = GPR(kernel=Matern())
        logging.debug("Fitting GP to\nx: {}\ny:{}".format(tX_, tY_))
        tgp.fit(tX_, tY_)  # fit the model
        tmu_star_t1_xy = get_mu_star(tgp)


        # Draw the left hand figure using the old gp on ax1
        draw_basic_figure(tgp=gp, title=r"$\hat{c}^{(t)}$", ax=ax1)

        logging.debug("Placing vertical on configuration: {}".format(tx_))

        ax1.scatter(
            tx_,
            ty_,
            color=colors['highlighted_observations'],
            marker='X',
            label=r"Hypothetical Observation $<\lambda, c(\lambda)>$",
            zorder=zorders["annotations_normal"]
        )

        ax1.legend().remove()

        # Draw the right hand figure using the hypothetical gp tgp on ax2
        draw_basic_figure(tgp=tgp, tx=tX_, tX_=tX_, tY_=tY_, title=r"$\hat{c}^{(t+1)}|_\lambda$",
                          highlight_datapoint=np.where(np.isclose(tX_, tx_))[0],
                          highlight_label=r"Hypothetical Observation $<\lambda, c(\lambda)>$", ax=ax2)


    def finishing_touches_parallel(ax1, ax2, filename=""):
        ax1.set_xlabel(labels['xlabel'])
        ax2.set_xlabel(labels['xlabel'])

        plt.tight_layout()
        if TOGGLE_PRINT:
            plt.savefig(f"{OUTPUT_DIR}/{filename}")
        else:
            plt.show()


    # ---------------------------------------
    # Draw look ahead 5

    fig, (ax1, ax2) = plt.subplots(1, 2, squeeze=True, figsize=(22, 9))
    draw_vertical_comparison(imaginary_lambda=5.0, ax1=ax1, ax2=ax2)
    finishing_touches_parallel(ax1=ax1, ax2=ax2, filename="look_ahead_5.pdf")

    # ---------------------------------------
    # Draw look ahead 6

    fig, (ax1, ax2) = plt.subplots(1, 2, squeeze=True, figsize=(22, 9))
    draw_vertical_comparison(imaginary_lambda=5.5, ax1=ax1, ax2=ax2)
    finishing_touches_parallel(ax1=ax1, ax2=ax2, filename="look_ahead_6.pdf")


    # ---------------------------------------
    # Draw look ahead 5

    fig, (ax1, ax2) = plt.subplots(1, 2, squeeze=True, figsize=(22, 9))
    draw_vertical_comparison(imaginary_lambda=3.5, ax1=ax1, ax2=ax2)
    finishing_touches_parallel(ax1=ax1, ax2=ax2, filename="look_ahead_7.pdf")


    # ---------------------------------------
    # Draw KG 1

    labels['gp_mean'] = r'Mean: $\mu^{(t)}(\cdot)$'

    fig, ax = draw_basic_figure(title="")
    perform_finishing_tasks(ax=ax, filename="kg_1.pdf", remove_legend=False)

    # ---------------------------------------
    # Draw kg 2
    fig, ax = draw_basic_figure(title="")

    boplot.highlight_configuration(mu_star_t_xy[0], lloc='bottom', ax=ax, disable_ticks=True)
    boplot.annotate_x_edge(label="%.2f" % mu_star_t_xy[0], xy=mu_star_t_xy, ax=ax, align='bottom',offset_param=1.5)
    boplot.highlight_output(mu_star_t_xy[1], label='', lloc='right', ax=ax, fontsize=30, disable_ticks=True)
    boplot.annotate_y_edge(label=r'${(\mu^*)}^{(t)}$', xy=mu_star_t_xy, align='right', ax=ax, yoffset=1.5)

    perform_finishing_tasks(ax=ax, filename="kg_2.pdf", remove_legend=True)

    # ---------------------------------------
    # Draw kg 3

    fig, ax = draw_basic_figure(tgp=gp2, tx=x, tX_=X2_, tY_=Y2_, title='',
                                highlight_datapoint=np.where(np.isclose(X2_, x_))[0],
                                highlight_label=r"Hypothetical Observation $<\lambda, c(\lambda)>$")

    perform_finishing_tasks(ax=ax, filename="kg_3.pdf", remove_legend=True)

    # ---------------------------------------
    # Draw kg 4


    fig, ax = draw_basic_figure(tgp=gp2, tx=x, tX_=X2_, tY_=Y2_, title='',
                                highlight_datapoint=np.where(np.isclose(X2_, x_))[0],
                                highlight_label=r"Hypothetical Observation $<\lambda, c(\lambda)>$")

    boplot.highlight_configuration(mu_star_t1_xy[0], lloc='bottom', ax=ax, disable_ticks=True)
    boplot.annotate_x_edge(label="%.2f" % mu_star_t1_xy[0], xy=mu_star_t1_xy, ax=ax, align='bottom', offset_param=1.5)

    boplot.highlight_output(mu_star_t1_xy[1], label='', lloc='right', ax=ax, fontsize=28)
    boplot.annotate_y_edge(label=r'${(\mu^*)}^{(t+1)}|_\lambda$', xy=mu_star_t1_xy, align='right', ax=ax, yoffset=1.5)

    perform_finishing_tasks(ax=ax, filename="kg_4.pdf", remove_legend=True)

    # ---------------------------------------
    # Draw kg 5

    fig, (ax1, ax2) = plt.subplots(1, 2, squeeze=True, figsize=(22, 9))
    draw_vertical_comparison(imaginary_lambda=x_.squeeze(), ax1=ax1, ax2=ax2)

    boplot.highlight_output(mu_star_t_xy[1], label='', lloc='right', ax=ax1, fontsize=30)
    boplot.annotate_y_edge(label='${(\mu^*)}^{(t)}$', xy=mu_star_t_xy, align='right', ax=ax1, yoffset=1.5)

    boplot.highlight_output(mu_star_t1_xy[1], label='', lloc='right', ax=ax2, fontsize=28)
    boplot.annotate_y_edge(label='${(\mu^*)}^{(t+1)}|_\lambda$', xy=mu_star_t1_xy, align='left', ax=ax2, yoffset=1.5)

    finishing_touches_parallel(ax1=ax1, ax2=ax2, filename="kg_5.pdf")

    return


def main(init_size, initial_design):
        visualize_look_ahead(
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
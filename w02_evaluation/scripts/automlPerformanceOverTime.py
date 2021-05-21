"""
========================================================
Optimize the hyperparameters of a support vector machine
========================================================
An example for the usage of SMAC within Python.
We optimize a simple SVM on the IRIS-benchmark.
Note: SMAC-documentation uses linenumbers to generate docs from this file.
"""
import json
import logging
from pathlib import Path
from typing import Union, Dict, Tuple, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ConfigSpace.conditions import InCondition
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter
from sklearn import svm, datasets
from sklearn.model_selection import cross_val_score

# Import ConfigSpace and different types of parameters
from smac.configspace import ConfigurationSpace
from smac.facade.smac_ac_facade import SMAC4AC
from smac.facade.smac_hpo_facade import SMAC4HPO
# Import SMAC-utilities
from smac.scenario.scenario import Scenario

plt.style.use(['ggplot', 'seaborn-talk'])


def generate_data(smac_class, n_runs=1, output_dir: Union[str, Path] = ".", dataset=None, runcount_limit=50):
    output_dir = Path(output_dir)

    if dataset is None:
        dataset = datasets.load_iris()

    def svm_from_cfg(cfg):
        """ Creates a SVM based on a configuration and evaluates it on the
        iris-dataset using cross-validation.
        Parameters:
        -----------
        cfg: Configuration (ConfigSpace.ConfigurationSpace.Configuration)
            Configuration containing the parameters.
            Configurations are indexable!
        Returns:
        --------
        A crossvalidated mean score for the svm on the loaded data2-set.
        """
        # For deactivated parameters, the configuration stores None-values.
        # This is not accepted by the SVM, so we remove them.
        cfg = {k: cfg[k] for k in cfg if cfg[k]}
        # We translate boolean values:
        cfg["shrinking"] = True if cfg["shrinking"] == "true" else False
        # And for gamma, we set it to a fixed value or to "auto" (if used)
        if "gamma" in cfg:
            cfg["gamma"] = cfg["gamma_value"] if cfg["gamma"] == "value" else "auto"
            cfg.pop("gamma_value", None)  # Remove "gamma_value"

        clf = svm.SVC(**cfg, random_state=None)

        scores = cross_val_score(clf, dataset.data, dataset.target, cv=5)
        return 1 - np.mean(scores)  # Minimize!

    # logger = logging.getLogger("SVMExample")
    logging.basicConfig(level=logging.INFO)  # logging.DEBUG for debug output

    # Build Configuration Space which defines all parameters and their ranges
    cs = ConfigurationSpace()

    # We define a few possible types of SVM-kernels and add them as "kernel" to our cs
    kernel = CategoricalHyperparameter("kernel", ["linear", "rbf", "poly", "sigmoid"], default_value="poly")
    cs.add_hyperparameter(kernel)

    # There are some hyperparameters shared by all kernels
    C = UniformFloatHyperparameter("C", 0.001, 1000.0, default_value=1.0)
    shrinking = CategoricalHyperparameter("shrinking", ["true", "false"], default_value="true")
    cs.add_hyperparameters([C, shrinking])

    # Others are kernel-specific, so we can add conditions to limit the searchspace
    degree = UniformIntegerHyperparameter("degree", 1, 5, default_value=3)  # Only used by kernel poly
    coef0 = UniformFloatHyperparameter("coef0", 0.0, 10.0, default_value=0.0)  # poly, sigmoid
    cs.add_hyperparameters([degree, coef0])
    use_degree = InCondition(child=degree, parent=kernel, values=["poly"])
    use_coef0 = InCondition(child=coef0, parent=kernel, values=["poly", "sigmoid"])
    cs.add_conditions([use_degree, use_coef0])

    # This also works for parameters that are a mix of categorical and values from a range of numbers
    # For example, gamma can be either "auto" or a fixed float
    gamma = CategoricalHyperparameter("gamma", ["auto", "value"], default_value="auto")  # only rbf, poly, sigmoid
    gamma_value = UniformFloatHyperparameter("gamma_value", 0.0001, 8, default_value=1)
    cs.add_hyperparameters([gamma, gamma_value])
    # We only activate gamma_value if gamma is set to "value"
    cs.add_condition(InCondition(child=gamma_value, parent=gamma, values=["value"]))
    # And again we can restrict the use of gamma in general to the choice of the kernel
    cs.add_condition(InCondition(child=gamma, parent=kernel, values=["rbf", "poly", "sigmoid"]))

    # Scenario object
    for i in range(n_runs):
        scenario = Scenario({"run_obj": "quality",  # we optimize quality (alternatively runtime)
                             "runcount-limit": runcount_limit,
                             # max. number of function evaluations; for this example set to a low number
                             "cs": cs,  # configuration space
                             "deterministic": "true",
                             "limit_resources": "false",
                             "output_dir": str((output_dir / smac_class.__name__ / f"{i:02d}").absolute())
                             })

        # Example call of the function
        # It returns: Status, Cost, Runtime, Additional Infos
        # def_value = svm_from_cfg(cs.get_default_configuration())
        # print(f"Default Value: {def_value:.2f}")
        #
        # Optimize, using a SMAC-object
        smac = smac_class(scenario=scenario, rng=None,
                          tae_runner=svm_from_cfg)

        incumbent = smac.optimize()
        #
        inc_value = svm_from_cfg(incumbent)
        #
        # print(f"Optimized Value: {inc_value:.2f}")
        #
        # # We can also validate our results (though this makes a lot more sense with instances)
        smac.validate(config_mode='inc',  # We can choose which configurations to evaluate
                      # instance_mode='train+test',  # Defines what instances to validate
                      repetitions=100,  # Ignored, unless you set "deterministic" to "false" in line 95
                      n_jobs=1)  # How many cores to use in parallel for optimization


def read_trajectory(folder: Path, *,
                    x_key="evaluations", y_key="cost", skip_rows=1) -> Dict[str, Dict[str, List[Tuple[float, float]]]]:
    if not folder.exists():
        raise FileNotFoundError(f"Cannot find {folder}. Maybe you have to generate the data first?")
    res = {}
    for algorithm_dir in folder.iterdir():
        if not algorithm_dir.is_dir():
            continue

        algorithm_name = algorithm_dir.name

        algorithm_result = {}
        for iteration_dir in algorithm_dir.iterdir():
            for run_dir in iteration_dir.iterdir():
                run_name = f"{iteration_dir.name}_{run_dir.name}"
                run_data = []
                traj_file = run_dir / "traj.json"
                with traj_file.open("r") as d:
                    for line in list(d)[skip_rows:]:
                        line_data = json.loads(line)
                        run_data.append((line_data[x_key], line_data[y_key]))

                algorithm_result[run_name] = run_data
        res[algorithm_name] = algorithm_result
    return res


def plot_all(data: Dict[str, Dict[str, List[Tuple[float, float]]]], *,
             group_color=True, step=False,
             log_x=False, log_y=False,
             save_fig_path: Path = None, show_plots=True):
    for algorithm, c in zip(data, ["red", "blue"]):
        if group_color:
            color = c
            plt.plot([], [], color=c, label=algorithm)
        else:
            color = None
        for run, run_data in data[algorithm].items():
            if group_color:
                label = None
            else:
                label = f"{algorithm}_{run}"
            x, y = list(zip(*run_data))

            if step:
                plt.step(x, y, label=label, color=color, where="post")
            else:
                plt.plot(x, y, label=label, color=color)

    if log_x:
        plt.xscale("log")
    if log_y:
        plt.yscale("log")

    plt.ylim(bottom=0)
    plt.xlabel("cpu time")
    plt.ylabel("incumbent cost")
    plt.legend()
    plt.tight_layout()

    if save_fig_path is not None:
        plt.savefig(save_fig_path)
    if show_plots:
        plt.show()

    plt.clf()


def fill_trajectory(time_list, performance_list):
    """
    modified:
    https://github.com/automl/plotting_scripts/blob/master/plottingscripts/utils/merge_test_performance_different_times.py
    """
    if len(performance_list) < 2:
        return np.array(performance_list), np.array(time_list).flatten()

    frame_dict = {}
    for c, (p, t) in enumerate(zip(performance_list, time_list)):
        if len(p) != len(t):
            raise ValueError(f"({c}) Array length mismatch: {len(p)} != {len(t)}")
        frame_dict[str(c)] = pd.Series(data=p, index=t)

    merged = pd.DataFrame(frame_dict)
    indices = merged.index.values
    fill_nan_row = merged.loc[indices.max()].copy()
    for idx in indices[::-1]:
        row = merged.loc[idx]
        nan_values = pd.isna(row)
        fill_nan_row[-nan_values] = row[-nan_values]
        merged.loc[idx] = fill_nan_row
    merged.fillna(np.min(merged), inplace=True)

    if not np.isfinite(merged.to_numpy()).all():
        raise ValueError("\nCould not merge lists, because \n"
                         "\t(a) one list is empty?\n"
                         "\t(b) the lists do not start with the same times and"
                         " replace_nan is not set?\n"
                         "\t(c) replace_nan is not set and there are non valid "
                         "numbers in the list\n"
                         "\t(d) any other reason.")
    return merged


def group_values(data: Dict[str, List[Tuple[float, float]]],
                 main_line="mean", bounds=None,
                 ) -> Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    times = []
    values = []
    for d in data:
        t, v = list(zip(*d))
        times.append(list(t))
        values.append(list(v))
    merged = fill_trajectory(times, values)

    time_ = merged.index.values

    # Main line
    if main_line == "mean":
        _main_line = merged.mean(axis=1).to_numpy()
    elif main_line == "median":
        _main_line = merged.median(axis=1).to_numpy()
    elif main_line is None:
        _main_line = None
    else:
        raise ValueError(f"Could not identify `{main_line}` as identifier for main_line")

    # Bounds
    if bounds == "stderr":
        # https://en.wikipedia.org/wiki/Standard_error
        mean = merged.mean(axis=1).to_numpy()
        stdev = merged.std(axis=1, ddof=1).to_numpy()
        stderr = stdev / np.sqrt(merged.shape[1])
        _bounds = mean - stderr, mean + stderr
    elif bounds == "stdev":
        mean = merged.mean(axis=1).to_numpy()
        stdev = merged.std(axis=1).to_numpy()
        _bounds = mean - stdev, mean + stdev
    elif bounds == "percentile":
        _bounds = np.percentile(merged.to_numpy(), [25, 75], axis=1)
    elif bounds is None:
        _bounds = (None, None)
    else:
        raise ValueError(f"Could not identify `{bounds}` as identifier for bounds")

    return time_, _main_line, _bounds


def plot_grouped(data: Dict[str, Dict[str, List[Tuple[float, float]]]], *,
                 step=True, log_x=False, log_y=False,
                 main_line="mean", bound_lines="stdev",
                 save_fig_path=None, show_plots=True):
    assert main_line in ["mean", "median"]
    assert bound_lines in ["stdev", "stderr", "percentile"]

    for algorithm, c in zip(data, [(1., 0, 0), (0, 0, 1.)]):
        # Add alpha
        color = tuple(list(c) + [1.])
        color_fill = tuple(list(c) + [0.5])

        # Plot legend
        plt.plot([], [], color=color, label=algorithm)

        # Get grouped data
        x, y, (bounds_low, bounds_high) = group_values(data[algorithm].values(), main_line=main_line,
                                                       bounds=bound_lines)

        # Plot boundary lines
        if bounds_low is not None and bounds_high is not None:
            plt.fill_between(x, bounds_low, bounds_high, color=color_fill, step="post" if step else None)

        # Plot main line
        if y is not None:
            if step:
                plt.step(x, y, label=None, color=color, where="post")
            else:
                plt.plot(x, y, label=None, color=color)

    # Adjust aces
    if log_x:
        plt.xscale("log")
    if log_y:
        plt.yscale("log")

    plt.title(f"{main_line} + {bound_lines}")
    plt.xlabel("cpu time")
    plt.ylabel("incumbent cost")
    plt.ylim(bottom=0)
    plt.legend()
    plt.tight_layout()
    if save_fig_path is not None:
        plt.savefig(save_fig_path)
    if show_plots:
        plt.show()

    plt.clf()


if __name__ == '__main__':
    DATA_FOLDER = Path("data_runcount_limit_500/")
    PLOT_FOLDER = Path("../plots")
    GENERATE_DATA = False
    PLOT_DATA = True
    SHOW_PLOTS = False

    if GENERATE_DATA:
        generate_data(SMAC4HPO, 10, output_dir=DATA_FOLDER, runcount_limit=500)
        generate_data(SMAC4AC, 10, output_dir=DATA_FOLDER, runcount_limit=500)

    if not PLOT_DATA:
        exit(0)

    trajectory_data = read_trajectory(DATA_FOLDER)
    all_data = {"SMAC4HPO": trajectory_data["SMAC4HPO"], "SMAC4AC": trajectory_data["SMAC4AC"]}  # Change order of plots
    smac4hpo_data = {"SMAC4HPO": trajectory_data["SMAC4HPO"]}
    PLOT_FOLDER.mkdir(exist_ok=True, parents=True)

    plot_all(smac4hpo_data,
             save_fig_path=PLOT_FOLDER / "4_smac4hpo.png", show_plots=SHOW_PLOTS)
    plot_all(smac4hpo_data, step=True,
             save_fig_path=PLOT_FOLDER / "5_smac4hpo_step.png", show_plots=SHOW_PLOTS)
    plot_all(smac4hpo_data, step=True, log_x=True,
             save_fig_path=PLOT_FOLDER / "6_1_smac4hpo_step_log_x.png", show_plots=SHOW_PLOTS)
    plot_all(smac4hpo_data, step=True, log_y=True,
             save_fig_path=PLOT_FOLDER / "6_2_smac4hpo_step_log_y.png", show_plots=SHOW_PLOTS)
    plot_all(smac4hpo_data, step=True, log_x=True, log_y=True,
             save_fig_path=PLOT_FOLDER / "6_3_smac4hpo_step_log_x_y.png", show_plots=SHOW_PLOTS)

    plot_grouped(smac4hpo_data, step=True, main_line="mean", bound_lines="stdev",
                 save_fig_path=PLOT_FOLDER / "8_1_smac4hpo_mean_stdev.png", show_plots=SHOW_PLOTS)
    plot_grouped(smac4hpo_data, step=True, main_line="mean", bound_lines="stderr",
                 save_fig_path=PLOT_FOLDER / "8_2_smac4hpo_mean_stderr.png", show_plots=SHOW_PLOTS)
    plot_grouped(smac4hpo_data, step=True, main_line="median", bound_lines="percentile",
                 save_fig_path=PLOT_FOLDER / "8_3_smac4hpo_median_percentile.png", show_plots=SHOW_PLOTS)

    plot_grouped(all_data, step=True, main_line="mean", bound_lines="stdev",
                 save_fig_path=PLOT_FOLDER / "9_1_compare_median_percentile.png", show_plots=SHOW_PLOTS)
    plot_grouped(all_data, step=True, main_line="mean", bound_lines="stderr",
                 save_fig_path=PLOT_FOLDER / "9_2_compare_mean_stderr.png", show_plots=SHOW_PLOTS)
    plot_grouped(all_data, step=True, main_line="median", bound_lines="percentile",
                 save_fig_path=PLOT_FOLDER / "9_3_compare_median_percentile.png", show_plots=SHOW_PLOTS)

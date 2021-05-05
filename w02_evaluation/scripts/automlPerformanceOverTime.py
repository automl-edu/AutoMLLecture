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


# We load the iris-dataset (a widely used benchmark)

def generate_data(smac_class, n_runs=1, output_dir: Union[str, Path] = ".", dataset=None):
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

        clf = svm.SVC(**cfg, random_state=42)

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
                             "runcount-limit": 50,
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
        smac = smac_class(scenario=scenario, rng=np.random.RandomState(42),
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


def read_trajectory(folder: Path) -> Dict[str, Dict[str, List[Tuple[float, float]]]]:
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
                    for line in d:
                        line_data = json.loads(line)
                        run_data.append((line_data["cpu_time"], line_data["incumbent"]["C"]))

                algorithm_result[run_name] = run_data
        res[algorithm_name] = algorithm_result
    return res


def plot_all(data: Dict[str, Dict[str, List[Tuple[float, float]]]], group_color=True, step=False, log_x=False,
             log_y=False):
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
                plt.step(x, y, label=label, color=color)
            else:
                plt.plot(x, y, label=label, color=color)

    if log_x:
        plt.xscale("log")
    if log_y:
        plt.yscale("log")

    plt.legend()
    plt.tight_layout()
    plt.show()


def group_values(data:List[Tuple[float, float]]):
    pass


def plot_grouped(data: Dict[str, Dict[str, List[Tuple[float, float]]]],
                 step=True, log_x=False, log_y=False,
                 main_line="mean", range_lines="stddev"):
    for algorithm, c in zip(data, ["red", "blue"]):
        plt.plot([], [], color=c, label=algorithm)
        x,y = group_values(data[algorithm].values())

        if step:
            plt.step(x, y, label=None, color=color)
        else:
            plt.plot(x, y, label=None, color=color)

    if log_x:
        plt.xscale("log")
    if log_y:
        plt.yscale("log")

    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    folder = Path("data2/")
    GENERATE_DATA = False
    PLOT_DATA = True
    if GENERATE_DATA:
        generate_data(SMAC4HPO, 10, output_dir=folder)
        generate_data(SMAC4AC, 10, output_dir=folder)

    if not PLOT_DATA:
        exit(0)

    trajectory_data = read_trajectory(folder)
    plot_all(trajectory_data)
    plot_all(trajectory_data, step=True)
    plot_all(trajectory_data, step=True, log_x=True)
    plot_all(trajectory_data, step=True, log_y=True)
    plot_all(trajectory_data, step=True, log_x=True, log_y=True)

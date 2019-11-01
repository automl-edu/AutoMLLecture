import numpy as np
np.warnings.filterwarnings('ignore')

import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics

# Load data
X, y = sklearn.datasets.load_digits(return_X_y=True)
X_train, X_valid, y_train, y_valid = \
        sklearn.model_selection.train_test_split(X, y, random_state=1)

from sklearn.neural_network import MLPClassifier


def eval_mlp(config, seed):
    mlp = MLPClassifier(hidden_layer_sizes=(128, 64,),
                        activation='relu',
                        solver='sgd',
                        alpha=config['alpha'],
                        learning_rate='constant',
                        learning_rate_init=config['lrate'],
                        momentum=config['momentum'],
                        warm_start=False,
                        random_state=seed)

    losses = []
    classes = np.unique(y_train)
    for i in range(1, 100):
        mlp.partial_fit(X_train, y_train, classes=classes)
        losses.append(mlp.loss_)

    #print(losses)
    return 1 - mlp.score(X_valid, y_valid), {'losses':losses, 'train': mlp.score(X_train, y_train)}

if True:

    def collect_data(config, n, name):
        losses = []
        test_scores = []
        train_scores = []
        for i in range (1, n):
            l = eval_mlp(config, seed=i)
            losses.append(l[1]['losses'])
            test_scores.append(l[0])
            train_scores.append(l[1]['train'])
        losses = np.array(losses)
        np.savetxt('%s_learning_curves.txt'  %(name), losses)
        np.savetxt('%s_test_scores.txt' %(name), np.array(test_scores))
        np.savetxt('%s_train_scores.txt' %(name), np.array(train_scores))

    collect_data({
                'alpha':0.0001,
                'lrate':0.001,
                'momentum':0.9
                }, 100, "mlp1")
    collect_data({
                'alpha':0.0001,
                'lrate':0.01,
                'momentum':0.9
                }, 100, "mlp2")

# ConfigSpace
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter

cs = ConfigurationSpace()
alpha = UniformFloatHyperparameter('alpha', lower=0.000001, upper=1,
                                   default_value=0.0001, log=True)
lrate = UniformFloatHyperparameter('lrate', lower=0.000001, upper=1,
                                  default_value=0.001, log=True)
momentum = UniformFloatHyperparameter('momentum', lower=0.5, upper=1,
                                      default_value=0.9)
cs.add_hyperparameters([alpha, lrate, momentum])

print(cs)

#def_conf = cs.get_default_configuration()
#eval_mlp(def_conf, seed=1)

if True:
    results = []
    for c in cs.sample_configuration(100):
        test_accs = []
        for i in range(3):
            t, a = eval_mlp(c, seed=1)
            test_accs.append(1 - t)
        test_accs = np.mean(test_accs)
        results.append(test_accs)
    np.savetxt(results, "hundred_configs_test_acc.txt")


if False:
    # Run SMAC
    from smac.scenario.scenario import Scenario
    from smac.facade.smac_facade import SMAC
    from smac.facade.borf_facade import BORF

    scenario = Scenario({"run_obj": "quality",   # we optimize quality (alternatively runtime)
                         "runcount-limit": 300,  # maximum function evaluations
                         "cs": cs,               # configuration space
                         "deterministic": "false"
                         })

    smac = BORF(scenario=scenario,
                rng=np.random.RandomState(42),
                tae_runner=eval_mlp)

    # at most 4 runs for non-deterministic scenarios
    smac.solver.intensifier.maxR = 4

    incumbent = smac.optimize()




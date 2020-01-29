#!/bin/python

import os

from sklearn import datasets
from sklearn import metrics
#from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn import cross_validation
from sklearn.metrics import accuracy_score

import struct
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import itertools
import sys
import random

from matplotlib.pyplot import tight_layout, figure, subplots_adjust, subplot, savefig, show
import matplotlib.gridspec
import plot_util


def plot_optimization_trace(times, performance_list, title, name_list,
                            log=False, save="", y_min=0, y_max=500,
                            x_min=0, x_max=None, aggregate="median",
                            markers='o', one_color=False):
    '''
    plots a median optimization trace based one time array
    '''
    
    if one_color:
        colors = itertools.cycle(["#e41a1c"])
    else:
        colors = itertools.cycle(["#e41a1c",    # Red
                              "#377eb8",    # Blue
                              "#4daf4a",    # Green
                              "#984ea3",    # Purple
                              "#ff7f00",    # Orange
                              "#ffff33",    # Yellow
                              "#a65628",    # Brown
                              "#f781bf",    # Pink
                              "#999999"])   # Grey
    linestyles = '-'
    size = 1

    # Set up figure
    ratio = 5
    gs = matplotlib.gridspec.GridSpec(ratio, 1)
    fig = figure(1, dpi=100)
    fig.suptitle(title, fontsize=16)
    ax1 = subplot(gs[0:ratio, :])
    ax1.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    
    platform_c_maxint = 2 ** (struct.Struct('i').size * 8 - 1) - 1

    auto_y_min = platform_c_maxint
    auto_y_max = -platform_c_maxint
    auto_x_min = platform_c_maxint

    for idx, performance in enumerate(performance_list):
        color = next(colors)
        # Get mean and std
        if log:
            performance = np.log10(performance)

        if aggregate == "median":
            m = np.median(performance, axis=0)
            upper_quartile = np.percentile(performance, q=75, axis=0)
            lower_quartile = np.percentile(performance, q=25, axis=0)
        else:
            m = np.mean(performance, axis=0)
            std_ = np.std(performance, axis=0)
            upper_quartile = m + std_
            lower_quartile = m - std_
        # Plot mean and std
        ax1.fill_between(times, lower_quartile, upper_quartile,
                         facecolor=color, alpha=0.3, edgecolor=color)
        if name_list:
            ax1.plot(times, m, color=color, linewidth=size,
                 linestyle=linestyles, marker=markers, label=name_list[idx])
        else:
            ax1.plot(times, m, color=color, linewidth=size,
                 linestyle=linestyles, marker=markers)

        # Get limits
        # For y_min we always take the lowest value
        auto_y_min = min(min(lower_quartile[x_min:]), auto_y_min)

        # For y_max we take the highest value after the median/quartile starts
        # to change
        init_per = m[0]
        init_up = upper_quartile[0]
        init_lo = lower_quartile[0]
        init_idx = 0
        # Find out when median/quartile changes
        while init_idx < len(m) and init_per == m[init_idx] and \
                init_up == upper_quartile[init_idx] and \
                init_lo == lower_quartile[init_idx]:
            # stop when median/quartile changes
            init_idx += 1

        # Found the first change, but show two more points on the left side
        init_idx = max(0, init_idx - 3)
        if init_idx >= 0:
            # median stays the same for > 1 evaluations
            auto_x_min = min(times[init_idx], auto_x_min)

        from_ = max(init_idx, x_min)
        auto_y_max = max(max(upper_quartile[from_:]), auto_y_max)
    auto_x_max = times[-1]

    # Describe axes
    if log:
        ax1.set_ylabel("log10(Performance)")
    else:
        ax1.set_ylabel("Accuracy")
    ax1.set_xlabel("#Iterations")

    if name_list:
        leg = ax1.legend(loc='best', fancybox=True)
        leg.get_frame().set_alpha(0.5)

    # Set axes limits
    # ax1.set_xscale("log")
    if y_max is None and y_min is not None:
        ax1.set_ylim([y_min, auto_y_max + 0.01 * abs(auto_y_max - y_min)])
    elif y_max is not None and y_min is None:
        ax1.set_ylim([auto_y_min - 0.01 * abs(auto_y_max - y_min), y_max])
    elif y_max > y_min and y_max is not None and y_min is not None:
        ax1.set_ylim([y_min, y_max])
    else:
        ax1.set_ylim([auto_y_min - 0.01 * abs(auto_y_max - auto_y_min),
                      auto_y_max + 0.01 * abs(auto_y_max - auto_y_min)])

    if x_max is None and x_min is not None:
        ax1.set_xlim(
            [x_min - 0.1 * abs(x_min), auto_x_max + 0.1 * abs(auto_x_max)])
    elif x_max is not None and x_min is None:
        ax1.set_xlim(
            [auto_x_min - 0.1 * abs(auto_x_min), x_max + 0.1 * abs(x_max)])
    elif x_max > x_min and x_max is not None and x_min is not None:
        ax1.set_xlim([x_min, x_max])
    else:
        ax1.set_xlim(
            [auto_x_min - 0.1 * abs(auto_x_min), auto_x_max + 0.1 * abs(auto_x_max)])

    # Save or show
    tight_layout()
    subplots_adjust(top=0.85)
    if save != "":
        savefig(save, dpi=100, facecolor='w', edgecolor='w',
                orientation='portrait', papertype=None, format=None,
                transparent=False, pad_inches=0.1)
    else:
        show()

MAX_K = 20
STEPS = 1

SET_ID = 4
REPS = 100

def get_data(crit="gini", max_features="auto"):

    if not os.path.isfile("rf_plots/rf_test_%s.np.npy" %(crit)):
        # load datasets
        with open("../ml_exercise4/set%d/x_train.np" % (SET_ID)) as fp:
            X_train = np.loadtxt(fp)
        with open("../ml_exercise4/set%d/y_train.np" % (SET_ID)) as fp:
            y_train = np.loadtxt(fp)
    
        with open("../ml_exercise4/set%d/x_test.np" % (SET_ID)) as fp:
            X_test = np.loadtxt(fp)
        with open("../ml_exercise4/set%d/y_test.np" % (SET_ID)) as fp:
            y_test = np.loadtxt(fp)
    
        times = []
        test_perfomances_eta = []
        train_performances_eta = []
        for rep in range(REPS):
            test_scores = []
            train_scores = []
            seed = random.randint(1, 100000000)
            
            model = RandomForestClassifier(n_estimators=1, warm_start=True, random_state=seed,
                                      criterion=crit, max_features=max_features)
            
            for k in range(1, MAX_K, STEPS):
    
                model.set_params(warm_start=True, n_estimators=k)
                model.fit(X_train, y_train)
    
                # make predictions
                y_pred_test = model.predict(X_test)
                y_pred_train = model.predict(X_train)
    
                # summarize the fit of the model
    
                #print(k, np.mean(score))
                test_scores.append(accuracy_score(y_test, y_pred_test))
                train_scores.append(accuracy_score(y_train, y_pred_train))
    
            times.append(range(1, MAX_K, STEPS))
            test_perfomances_eta.append(test_scores)
            train_performances_eta.append(train_scores)
    
        np.save("rf_plots/rf_test_a%s.np" %(crit), np.array(test_perfomances_eta))
        np.save("rf_plots/rf_train_a%s.np" %(crit), np.array(train_performances_eta))
    
    else:
    
        test_perfomances_eta = np.load("rf_plots/rf_test_a%s.np.npy" %(crit))
        train_performances_eta = np.load("rf_plots/rf_train_a%s.np.npy" %(crit))
        
    return test_perfomances_eta, train_performances_eta


def get_rtd_data (runs, quality=0.76):
    rtd = []
    for r in runs:
        for idx, d in enumerate(r):
            if d >= quality:
                rtd.append(idx+1)
                break
    return rtd

def boxplot(x1, x2, save_name, labels=[]):
    fig, ax1 = plt.subplots()
    ax1.boxplot([np.array(x1), np.array(x2)])
    #plt.yscale('log')
    
    if labels:
        xtickNames = plt.setp(ax1, xticklabels=labels)
        plt.setp(xtickNames, rotation=25, fontsize=12)
    
    plt.ylabel("#Iterations")
    
    plt.savefig("rf_plots/%s_boxplot.pdf" %(save_name), format="pdf")
    
def get_cdf_plot(baseline, configured, cutoff, save=""):
    '''
        generate cdf plot
    '''
    #TODO: READ cutoff from somewhere
    
    colors = itertools.cycle(["#e41a1c",    # Red
                      "#377eb8",    # Blue
                      "#4daf4a",    # Green
                      "#984ea3",    # Purple
                      "#ff7f00",    # Orange
                      "#ffff33",    # Yellow
                      "#a65628",    # Brown
                      "#f781bf",    # Pink
                      "#999999"])   # Grey
    
    user_fontsize=20
    
    font = {'size'   : user_fontsize}

    matplotlib.rc('font', **font)
    
    gs = matplotlib.gridspec.GridSpec(1, 1)
    
    fig = plt.figure()
    ax1 = plt.subplot(gs[0:1, :])
    
    #remove timeouts
    #baseline = filter(lambda x: True if x < cutoff else False, baseline)
    #configured = filter(lambda x: True if x < cutoff else False, configured)
    
    def get_x_y(data):
        b_x, b_y, i_s = [], [], 0
        for i, x in enumerate(np.sort(data)):
            b_x.append(x)
            if x < cutoff:
                b_y.append(float(i) /len(data))
                i_s = i
            else: 
                b_y.append(float(i_s) /len(data))
        return b_x, b_y
                
    #print(get_x_y(baseline)[1])
    #print(get_x_y(baseline)[0])
    ax1.step(get_x_y(baseline)[0], get_x_y(baseline)[1], label="criterion=gini", color=next(colors))
    ax1.step(get_x_y(configured)[0], get_x_y(configured)[1], label="criterion=entropy", color=next(colors))

    ax1.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax1.set_xlabel("#Iterations")
    ax1.set_ylabel("P(x<t)")
    #ax1.set_ylim([0,cutoff])
    ax1.set_xscale('log')

    ax1.legend(loc='upper left')

    out_file = os.path.join("rf_plots", "%s_cdf.png" %(save))
        
    plt.savefig(out_file, dpi=100, facecolor='w', edgecolor='w',
                orientation='portrait', papertype=None, format=None,
                transparent=False, pad_inches=0.02, bbox_inches='tight')

def exact_mc_perm_test(xs, ys, nmc):
    ''' source: http://stackoverflow.com/questions/24795535/pythons-implementation-of-permutation-test-with-permutation-number-as-input '''
    n, k = len(xs), 0
    diff = np.mean(xs) - np.mean(ys)
    zs = np.concatenate([xs, ys])
    for j in range(nmc):
        np.random.shuffle(zs)
        k += diff < np.mean(zs[:n]) - np.mean(zs[n:])
    print(k)
    return k / float(nmc)
    
test_perfomances_gini, train_performances_gini = get_data(crit="gini")
test_perfomances_entropy, train_performances_entropy = get_data(crit="entropy", max_features=None)
#test_perfomances_def, train_performances_def = get_data(learning_rate="optimal", eta0=0.)
    

plot_optimization_trace(np.array(range(1, MAX_K, STEPS)), np.array(
    [test_perfomances_gini]), "", ["SGD"], aggregate="median", save="rf_plots/test_median" )
plot_optimization_trace(np.array(range(1, MAX_K, STEPS)), np.array(
    [train_performances_gini]), "", ["SGD"], aggregate="median", save="rf_plots/train_median")

plot_optimization_trace(np.array(range(1, MAX_K, STEPS)), np.array(
    [test_perfomances_gini]), "", ["SGD"], aggregate="mean", save="rf_plots/test_mean")
plot_optimization_trace(np.array(range(1, MAX_K, STEPS)), np.array(
    [train_performances_gini]), "", ["SGD"], aggregate="mean", save="rf_plots/train_mean")

plot_optimization_trace(np.array(range(1, MAX_K, STEPS)), np.array(
    [[t,t] for t in train_performances_gini]), "", [], aggregate="mean", save="rf_plots/train_all", markers='', one_color=True)
plot_optimization_trace(np.array(range(1, MAX_K, STEPS)), np.array(
    [[t,t] for t in test_perfomances_gini]), "", [], aggregate="mean", save="rf_plots/test_all", markers='', one_color=True)

plot_optimization_trace(np.array(range(1, MAX_K, STEPS)), np.array(
    [[t,t] for t in test_perfomances_gini[:1]]), "", [], aggregate="mean", save="rf_plots/test_first", markers='', one_color=True)
plot_optimization_trace(np.array(range(1, MAX_K, STEPS)), np.array(
    [[t,t] for t in train_performances_gini[:1]]), "", [], aggregate="mean", save="rf_plots/train_first", markers='', one_color=True)


plot_optimization_trace(np.array(range(1, MAX_K, STEPS)), np.array(
    [test_perfomances_gini, test_perfomances_entropy]), "", ["RF criterion=gini","RF criterion=entropy"], aggregate="median", save="rf_plots/test_median_comparison" )
plot_optimization_trace(np.array(range(1, MAX_K, STEPS)), np.array(
    [train_performances_gini, train_performances_entropy]), "", ["RF criterion=gini","RF criterion=entropy"], aggregate="median", save="rf_plots/train_median_comparison")

plot_optimization_trace(np.array(range(1, MAX_K, STEPS)), np.array(
    [test_perfomances_gini, test_perfomances_entropy]), "", ["RF criterion=gini","RF criterion=entropy"], aggregate="mean", save="rf_plots/test_mean_comparison")
plot_optimization_trace(np.array(range(1, MAX_K, STEPS)), np.array(
    [train_performances_gini, train_performances_entropy]), "", ["RF criterion=gini","RF criterion=entropy"], aggregate="mean", save="rf_plots/train_mean_comparison")

#plot_optimization_trace(np.array(range(1, MAX_K, STEPS)), np.array(
#    [test_perfomances_gini, test_perfomances_entropy, test_perfomances_def]), "", ["RF criterion=gini","RF criterion=entropy", "SGD def"], aggregate="mean", save="rf_plots/test_mean_comparison_all")


rtd_gini = get_rtd_data(test_perfomances_gini)
rtd_entropy = get_rtd_data(test_perfomances_entropy)

boxplot(rtd_gini, rtd_entropy, save_name="gini_entropy", labels=["criterion=gini", "criterion=entropy"])

get_cdf_plot(np.array(rtd_gini,float),np.array(rtd_entropy,float), MAX_K, save="gini_entropy")

print("Permutation test (10 000 permutations): %f" %(exact_mc_perm_test(rtd_gini, rtd_entropy, 100000)))
print("Permutation test (10 000 permutations): %f" %(exact_mc_perm_test(rtd_entropy, rtd_gini, 100000)))

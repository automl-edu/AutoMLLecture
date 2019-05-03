import matplotlib.pyplot as plt
import numpy as np

from plottingscripts.plotting.plot_methods import plot_optimization_trace_mult_exp

mlp1_lcurve = np.loadtxt("mlp1_learning_curves.txt")
mlp2_lcurve = np.loadtxt("mlp2_learning_curves.txt")


fig = plot_optimization_trace_mult_exp(time_list=[list(range(1,100))],
                                 performance_list=[[mlp1_lcurve[1,:]]],
                                 name_list=["1"],
                                 ylabel="Error",
                                 xlabel="#Epochs",
                                properties = {'markersize': 0,
                                              'legendlocation':None},
                                    y_max=1.
                                    )
fig.savefig("one_learning_curve.jpg")


fig = plot_optimization_trace_mult_exp(time_list=[list(range(1,100)) for _ in range(0,10)],
                                 performance_list=[[mlp2_lcurve[i,:]] for i in range(0,10)],
                                 name_list=["1"]*10,
                                 ylabel="Error",
                                 xlabel="#Epochs",
                                properties = {'markersize': 0,
                                              'legendlocation':None},
                                y_max=1.
                                    )
fig.savefig("ten_learning_curves.jpg")

fig = plot_optimization_trace_mult_exp(time_list=[list(range(1,100))*1],
                                 performance_list=[mlp1_lcurve[:,:]],
                                 name_list=["1"],
                                 ylabel="Error",
                                 xlabel="#Epochs",
                                properties = {'markersize': 0,
                                              'legendlocation':None},
                                y_max=1.
                                    )
fig.savefig("hundred_agg_learning_curves.jpg")

fig = plot_optimization_trace_mult_exp(time_list=[list(range(1,100)), list(range(1,100))],
                                 performance_list=[mlp1_lcurve[:,:], mlp2_lcurve[:,:]],
                                 name_list=["1", "2"],
                                 ylabel="Error",
                                 xlabel="#Epochs",
                                properties = {'markersize': 0,
                                              'legendlocation':None},
                                y_max=1.
                                    )
fig.savefig("compare_learning_curves.jpg")

from plottingscripts.plotting.scatter import plot_scatter_plot

mlp1_testscores = np.loadtxt("mlp1_test_scores.txt")
mlp1_trainscores = np.loadtxt("mlp1_train_scores.txt")
mlp2_testscores = np.loadtxt("mlp2_test_scores.txt")
mlp2_trainscores = np.loadtxt("mlp2_train_scores.txt")

fig, ax1 = plt.subplots()
ax1.scatter(mlp1_trainscores, 1- mlp1_testscores)
ax1.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
ax1.set_xlabel('Train Error')
ax1.set_ylabel('Test Error')
fig.savefig("mlp1_test_train_scatter.jpg")

fig, ax1 = plt.subplots()
ax1.scatter(mlp2_trainscores, 1- mlp2_testscores)
ax1.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
ax1.set_xlabel('Train Error')
ax1.set_ylabel('Test Error')
fig.savefig("mlp2_test_train_scatter.jpg")


# CDF plot
mlp1_test_sorted = np.sort((1 - mlp1_testscores).tolist())
mlp2_test_sorted = np.sort((1 - mlp2_testscores).tolist())
y = np.array(range(1,100)) / 99

fig, ax1 = plt.subplots()
ax1.step(mlp1_test_sorted,y)
ax1.set_xlabel('Test Error')
ax1.set_ylabel('P(L<X)')

fig.savefig("mlp1_test_ecdf.jpg")

ax1.step(mlp2_test_sorted,y)

fig.savefig("mlp12_test_ecdf.jpg")

# Boxplot

fig, ax1 = plt.subplots()
ax1.boxplot([1 - mlp1_testscores, 1 - mlp2_testscores])
ax1.set_ylabel('Test Error')
fig.savefig("mlp12_boxplot.jpg")

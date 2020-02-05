#!/usr/bin/env python

'''
plot_scatter.py -- generates scatter plots

@author: Katharina Eggensperger and Marius Lindauer
@license: GPLv3

'''

from argparse import ArgumentParser
import sys
import itertools

from matplotlib.pyplot import tight_layout, figure, subplots_adjust,\
    subplot, savefig, show, setp
import matplotlib.gridspec
import numpy as np
from matplotlib.ticker import FormatStrFormatter

import plot_util


def plot_scatter_plot(x_data, y_data, labels, title="", save="", debug=False,
                      min_val=None, max_val=1000, grey_factor=1, linefactors=None,
                      user_fontsize=20, dpi=100):
    '''
        method to generate a scatter plot 
        Args:
            x_data: numpy.array
                performance values of one algorithm
            y_data: numpy.array
                performance values of the other algorithm
            title: str 
                title of plot
            save: str
                save plot to disk -- if not set, show plot
            debug: bool
                some debug options
            min_val: float
                minimal value to plot
            max_val: float 
                maximal value to plot
            grey_factor: float
                grey factor of points with a speedup of less 2
            linefactors: list of floats
                factors of speedups
            user_fontsize: int
                font size
            dpi: int
                resolution
            
    '''
    
    regular_marker = 'x'
    timeout_marker = '+'
    grey_marker = '.'
    c_angle_bisector = "#e41a1c"  # Red
    c_good_points = "#999999"     # Grey
    c_other_points = "k"
    size = 1
    st_ref = "--"

    ticklabel_size = user_fontsize
    linefactor_size = user_fontsize - 2
    label_size = user_fontsize + 1

    #------
    # maximum_value: location for timeout points
    # max_val      : Initially user-defined timeout, then set to axes limit
    # time_out_val : location for timeout points
    # -----

    maximum_value = max_val

    # Colors
    ref_colors = itertools.cycle([  # "#e41a1c",    # Red
                                 "#377eb8",    # Blue
                                 "#4daf4a",    # Green
                                 "#984ea3",    # Purple
                                 "#ff7f00",    # Orange
                                 "#ffff33",    # Yellow
                                 "#a65628",    # Brown
                                 "#f781bf",    # Pink
                                 # "#999999",    # Grey
                                 ])

    # Set up figure
    ratio = 1
    gs = matplotlib.gridspec.GridSpec(ratio, 1)
    fig = figure(1, dpi=100)
    fig.suptitle(title, fontsize=16)
    ax1 = subplot(gs[0:ratio, :], aspect='equal')
    ax1.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)

    # set initial limits
    if min_val is not None:
        auto_min_val = min([min(x_data), min(y_data), min_val])
    else:
        auto_min_val = min([min(x_data), min(y_data)])
    auto_max_val = maximum_value
    timeout_val = maximum_value * 2 #+ 10**int((np.log10(max_val)))

    # Plot angle bisector and reference_lines
    out_up = auto_max_val
    out_lo = max(10**-6, auto_min_val)

    ax1.plot([out_lo, out_up], [out_lo, out_up], c=c_angle_bisector)

    if linefactors is not None:
        for f in linefactors:
            c = ref_colors.next()
            # Lower reference lines
            ax1.plot([f*out_lo, out_up], [out_lo, (1.0/f)*out_up], c=c,
                     linestyle=st_ref, linewidth=size*1.5)
            # Upper reference lines
            ax1.plot([out_lo, (1.0/f)*out_up], [f*out_lo, out_up], c=c,
                     linestyle=st_ref, linewidth=size*1.5)
            offset = 1.1
            if int(f) == f:
                lf_str = "%dx" % f
            else:
                lf_str = "%2.1fx" % f
            ax1.text((1.0/f)*out_up, out_up*offset, lf_str, color=c,
                     fontsize=linefactor_size)
            ax1.text(out_up*offset, (1.0/f)*out_up, lf_str, color=c,
                     fontsize=linefactor_size)


    #######
    #  Scatter
    grey_idx = list()
    timeout_x = list()
    timeout_y = list()
    timeout_both = list()
    rest_idx = list()
    for idx_x, x in enumerate(x_data):
        if x >= max_val > y_data[idx_x]:
            # timeout of x algo
            timeout_x.append(idx_x)
        elif y_data[idx_x] >= max_val > x:
            # timeout of y algo
            timeout_y.append(idx_x)
        elif y_data[idx_x] >= max_val and x >= max_val:
            # timeout of both algos
            timeout_both.append(idx_x)
        elif y_data[idx_x] < grey_factor*x and x < grey_factor*y_data[idx_x]:
            grey_idx.append(idx_x)
        else:
            rest_idx.append(idx_x)

    # Regular points
    if len(grey_idx) > 1:
        ax1.scatter(x_data[grey_idx], y_data[grey_idx], marker=grey_marker,
                    c=c_good_points)
    ax1.scatter(x_data[rest_idx], y_data[rest_idx], marker=regular_marker,
                c=c_other_points)

    # max_val lines
    ax1.plot([maximum_value, maximum_value], [auto_min_val, maximum_value],
             c=c_other_points, linestyle="--", zorder=0, linewidth=size)
    ax1.plot([auto_min_val, maximum_value], [maximum_value, maximum_value],
             c=c_other_points, linestyle="--", zorder=0, linewidth=size)

    # Timeout points
    ax1.scatter([timeout_val]*len(timeout_x), y_data[timeout_x],
                marker=timeout_marker, c=c_other_points)
    ax1.scatter([timeout_val]*len(timeout_both), [timeout_val]*len(timeout_both),
                marker=timeout_marker, c=c_other_points)
    ax1.scatter(x_data[timeout_y], [timeout_val]*len(timeout_y),
                marker=timeout_marker, c=c_other_points)

    # Plot timeout line
#    ax1.plot([timeout_val, timeout_val], [auto_min_val, timeout_val],
#             c=c_other_points, linestyle=":", zorder=0)
#    ax1.plot([auto_min_val, timeout_val], [timeout_val, timeout_val],
#             c=c_other_points, linestyle=":", zorder=0)

    if debug:
        # debug option
        ax1.scatter(x_data, y_data, marker="o", facecolor="", s=50,
                    label="original data")

    # Set axes scale and limits
    ax1.set_xscale("log")
    ax1.set_yscale("log")

    # Set axes labels
    ax1.set_xlabel(labels[0], fontsize=label_size)
    ax1.set_ylabel(labels[1], fontsize=label_size)

    if debug:
        # Plot legend
        leg = ax1.legend(loc='best', fancybox=True)
        leg.get_frame().set_alpha(0.5)

    # Save or show figure
    tight_layout()
    subplots_adjust(top=0.85)
    
    max_val = timeout_val*2
    auto_min_val *= 0.9
    ax1.set_autoscale_on(False)
    if max_val is not None and min_val is None:
        # User sets max val
        ax1.set_ylim([auto_min_val, max_val])
        ax1.set_xlim(ax1.get_ylim())
    elif max_val > min_val and max_val is not None and min_val is not None:
        # User sets both, min and max -val
        ax1.set_ylim([min_val, max_val])
        ax1.set_xlim(ax1.get_ylim())
    else:
        # User sets nothing
        ax1.set_xlim([auto_min_val, max_val])
        ax1.set_ylim(ax1.get_xlim())

    # Plot maximum value as tick
    if int(maximum_value) == maximum_value:
        maximum_value = int(maximum_value)
        maximum_str = r"$%d$" % maximum_value
    else:
        maximum_str = r"$%5.2f$" % maximum_value

    if int(np.log10(maximum_value)) != np.log10(maximum_value):
        # If we do not already have this ticklabel as a regular label
        ax1.text(ax1.get_ylim()[0] - 0.1 * np.abs(ax1.get_ylim()[0]),
                 maximum_value,
                 maximum_str,
                 horizontalalignment='right', verticalalignment="center",
                 fontsize=user_fontsize)
        ax1.text(maximum_value,
                 ax1.get_ylim()[0] - 0.1 * np.abs(ax1.get_ylim()[0]),
                 maximum_str,
                 horizontalalignment='center', verticalalignment="top",
                 fontsize=user_fontsize)

    # Plot 'timeout'
    ax1.text(ax1.get_xlim()[0] - 0.1 * np.abs(ax1.get_ylim()[0]),
             timeout_val,
             "timeout ", horizontalalignment='right',
             verticalalignment="center", fontsize=user_fontsize)
    ax1.text(timeout_val,
             ax1.get_ylim()[0] - 0.1 * np.abs(ax1.get_ylim()[0]),
             "timeout ",  horizontalalignment='center', verticalalignment="top",
             fontsize=user_fontsize, rotation=45)

    #########
    # Adjust ticks > max_val
    ax1.xaxis.set_ticks_position('bottom')
    ax1.yaxis.set_ticks_position('left')
    # major axes
    for tic in ax1.xaxis.get_major_ticks():
        if tic._loc > maximum_value:
            tic.tick1On = tic.tick2On = False
    for tic in ax1.yaxis.get_major_ticks():
        if tic._loc > maximum_value:
            tic.tick1On = tic.tick2On = False

    # minor axes
    for tic in ax1.xaxis.get_minor_ticks():
        if tic._loc > maximum_value:
            tic.tick1On = tic.tick2On = False
    for tic in ax1.yaxis.get_minor_ticks():
        if tic._loc > maximum_value:
            tic.tick1On = tic.tick2On = False

    # tick labels
    ticks_x = ax1.get_xticks()
    new_ticks_label = list()
    for l_idx in range(len(ticks_x)):
        if ticks_x[l_idx] < maximum_value:
            new_ticks_label.append(ticks_x[l_idx])
    ax1.set_xticklabels(new_ticks_label)  # , rotation=45)

    ticks_y = ax1.get_yticks()
    new_ticks_label = list()
    for l_idx in range(len(ticks_y)):
        if ticks_x[l_idx] < maximum_value:
            if 0 < ticks_x[l_idx] < 1:
                new_ticks_label.append(str(r"$10^{%d}$" % int(np.log10(ticks_x[l_idx]))))
            if 1 <= ticks_x[l_idx] < 1000:
                new_ticks_label.append(str(r"$%d^{ }$" % int(ticks_x[l_idx])))
            if 1000 <= ticks_x[l_idx]:
                new_ticks_label.append(str(r"$10^{%d}$" % int(np.log10(ticks_x[l_idx]))))
    ax1.set_yticklabels(new_ticks_label)  # , rotation=45)
    ax1.set_xticklabels(new_ticks_label)

    # Change fontsize for ticklabels
    setp(ax1.get_yticklabels(), fontsize=ticklabel_size)
    setp(ax1.get_xticklabels(), fontsize=ticklabel_size)
    tight_layout()
    if save != "":
        savefig(save, dpi=dpi, facecolor='w', edgecolor='w',
                orientation='portrait', papertype=None, format=None,
                transparent=False, pad_inches=0.02, bbox_inches='tight')
    else:
        show()


def main():
    prog = "python plot_scatter.py"
    description = "Plots performances of the best config at one time vs another in a scatter plot"

    parser = ArgumentParser(description=description, prog=prog)

    # General Options
    # parser.add_argument("-l", "--log", action="store_true", dest="log",
    #                     default=False, help="Plot on log scale")
    parser.add_argument("--max", dest="max", type=float,
                        default=1000, help="Maximum of both axes")
    parser.add_argument("--min", dest="min", type=float,
                        default=None, help="Minimum of both axes")
    parser.add_argument("-s", "--save", dest="save",
                        default="", help="Where to save plot instead of showing it?")
    parser.add_argument("--title", dest="title",
                        default="", help="Optional supertitle for plot")
    parser.add_argument("--greyFactor", dest="grey_factor", type=float,
                        default=1, help="If an algorithms is not greyFactor-times better"
                                        " than the other, show this point less salient, > 1")
    parser.add_argument("-v", "--verbose", dest="verbose", action="store_true", default=False,
                        help="Plot some debug info")
    parser.add_argument("-f", "--lineFactors", dest="linefactors",
                        default=None, help="Plot X speedup/slowdown, format 'X,..,X' (no spaces)")
    parser.add_argument("--time", dest="time", default=None,
                        help="Plot config at which time?, format 'time1,time2' ")
    parser.add_argument("--obj", dest="obj", default=None, required=True,
                        help="Path to validationObjectiveMatrix-traj-run-* file")
    parser.add_argument("--res", dest="res", required=True,
                        help="Path to validationResults-traj-run-* file")
    parser.add_argument("--minvalue", dest="minvalue", type=float,
                        help="Replace all values smaller than this",)
    parser.add_argument("--fontsize", dest="fontsize", type=int, default=20,
                        help="Use this fontsize for plotting",)

    args, unknown = parser.parse_known_args()

    if len(unknown) != 0:
        print "Wrong number of arguments"
        parser.print_help()
        sys.exit(1)

    if args.grey_factor < 1:
        print "A grey-factor lower than one makes no sense"
        parser.print_help()
        sys.exit(1)

    # Load validationResults
    res_header, res_data = plot_util.read_csv(args.res, has_header=True)
    av_times = [float(row[0]) for row in res_data]
    if args.time == None:
        # Print available times and quit
        print "Choose a time from"
        print "\n".join(["* %s" % i for i in av_times])
        sys.exit(0)
    time_arr = args.time.split(",")
    if len(time_arr) != 2 or (len(time_arr) == 2 and (time_arr[1] == "" or time_arr[0] == "")):
        print "Something wrong with %s, should be 'a,b'" % args.time
        print "Choose a time from"
        print "\n".join(["* %s" % i for i in av_times])
        sys.exit(0)
    time_1 = float(time_arr[0])
    time_2 = float(time_arr[1])

    # Now extract data
    config_1 = [int(float(row[len(res_header)-2].strip('"'))) for row in res_data if int(float(row[0])) == int(time_1)]
    config_2 = [int(float(row[len(res_header)-2].strip('"'))) for row in res_data if int(float(row[0])) == int(time_2)]
    if len(config_1) == 0 or len(config_2) == 0:
        print "Time int(%s) or int(%s) not found. Choose a time from:" % (time_1, time_2)
        print "\n".join(["* %s" % i for i in av_times])
        sys.exit(1)
    config_1 = config_1[0]
    config_2 = config_2[0]

    obj_header, obj_data = plot_util.read_csv(args.obj, has_header=True)
    head_template = '"Objective of validation config #%s"'
    idx_1 = obj_header.index(head_template % config_1)
    idx_2 = obj_header.index(head_template % config_2)

    data_one = np.array([float(row[idx_1].strip('"')) for row in obj_data])
    data_two = np.array([float(row[idx_2].strip('"')) for row in obj_data])

    print "Found %s points for config %d and %s points for config %d" % (str(data_one.shape), config_1, str(data_two.shape), config_2)

    linefactors = list()
    if args.linefactors is not None:
        linefactors = [float(i) for i in args.linefactors.split(",")]
        if len(linefactors) < 1:
            print "Something is wrong with linefactors: %s" % args.linefactors
            sys.exit(1)
        if min(linefactors) < 1:
            print "A line-factor lower than one makes no sense"
            sys.exit(1)
    if args.grey_factor > 1 and args.grey_factor not in linefactors:
        linefactors.append(args.grey_factor)

    label_template = 'Objective of validation config #%s, best at %s sec'
    
    # This might produce overhead for large .csv files
    times = [int(float(row[0])) for row in res_data]
    time_1 = res_data[times.index(int(time_1))][0]
    time_2 = res_data[times.index(int(time_2))][0]

    data_one = np.array([max(args.minvalue, i) for i in data_one])
    data_two = np.array([max(args.minvalue, i) for i in data_two])

    save = ""
    if args.save != "":
        save = args.save
        print "Save to %s" % args.save
    else:
        print "Show"
    plot_scatter_plot(x_data=data_one, y_data=data_two,
                      labels=[label_template % (config_1, str(time_1)),
                              label_template % (config_2, str(time_2))],
                      title=args.title, save=save,
                      max_val=args.max, min_val=args.min,
                      grey_factor=args.grey_factor, linefactors=linefactors,
                      user_fontsize=args.fontsize,
                      debug=args.verbose)


if __name__ == "__main__":
    main()

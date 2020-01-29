import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import itertools
import sys, os
import random

from matplotlib.pyplot import tight_layout, figure, subplots_adjust, subplot, savefig, show
import matplotlib.gridspec

def get_cdf_plot(baseline, cutoff, save=""):
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
    ax1.step(get_x_y(baseline)[0], get_x_y(baseline)[1], color=colors.next())

    ax1.grid(True, linestyle='-', which='major', color='lightgrey', alpha=0.5)
    ax1.set_xlabel("Score")
    ax1.set_ylabel("P(x<X)")
    ax1.set_xlim([0,cutoff])
    ax1.set_ylim([0,1])
    #ax1.set_xscale('log')

    ax1.legend(loc='upper left')

    out_file = os.path.join(".", "%s_cdf.png" %(save))
        
    plt.savefig(out_file, dpi=100, facecolor='w', edgecolor='w',
                orientation='portrait', papertype=None, format=None,
                transparent=False, pad_inches=0.02, bbox_inches='tight')
    
scores = [31.25,34,47.25,42.25,45.25,39.75,44.5,39,25.25,32.5,31.75,46.75,46.5,31.25,25]

get_cdf_plot(baseline=scores, cutoff=max(scores), save="scores")
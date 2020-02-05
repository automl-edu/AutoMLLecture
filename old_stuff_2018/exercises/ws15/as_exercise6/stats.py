#!/bin/python

import sys
import os
import arff
import numpy

def get_score(algo_runs):
    
    inst_algo_time = {}
    algos = set()
    for d in algo_runs["data"]:
        inst = d[0]
        algo = d[2]
        algos.add(algo)
        if d[3] < 5000 and d[4] == "ok":
            time = d[3]
        else:
            time = 5000*10
        inst_algo_time[inst] = inst_algo_time.get(inst,{})
        inst_algo_time[inst][algo] = time
        
        
    algos = list(algos)
    
    matrix = []
    for algo_d in inst_algo_time.itervalues():
        times = []
        for algo in algos:
            true_time = algo_d[algo]
            if true_time <= 5000:
                times.append(true_time)
            else:
                times.append(5000 *10)
        matrix.append(times)
        
    matrix = numpy.array(matrix)
    
    oracle = numpy.mean(numpy.min(matrix, axis=1))
    sb = numpy.min(numpy.mean(matrix, axis=0))
            
    print("Oracle Score: %.2f" %(oracle))
    print("SB Score: %.2f" %(sb))
                

indu = os.path.abspath("./SAT11-INDU")
rand = os.path.abspath("./SAT11-RAND")

for scen in [indu, rand]:
    print(scen)
    
    with open(scen+"/algorithm_runs.arff") as fp:
        algo_runs = arff.load(fp)
    
    get_score(algo_runs)
    
    
    
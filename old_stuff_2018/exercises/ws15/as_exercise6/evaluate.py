#!/bin/python

import sys
import os
import json
import arff
from subprocess import Popen, PIPE

def get_score(algo_runs, assignment, permutation):
    
    inst_algo_time = {}
    for d in algo_runs["data"]:
        inst = d[0]
        algo = d[2]
        if d[3] < 5000 and d[4] == "ok":
            time = d[3]
        else:
            time = 5000*10
        inst_algo_time[inst] = inst_algo_time.get(inst,{})
        inst_algo_time[inst][algo] = time
        
    par_sum = 0
    for algo_d in inst_algo_time.itervalues():
        time = 0
        solved = False
        for algo in permutation:
            true_time = algo_d[algo]
            slot = float(assignment[algo])
            if slot >= true_time:
                time += true_time 
                solved = True
                break
            else:
                time += slot
        if solved and time <= 5000:
            par_sum += time
        else:
            par_sum += 5000 * 10
            
    print("PAR10 Score: %.2f" %(par_sum/len(inst_algo_time)))
                

py = sys.argv[1]
cwd = os.getcwd()
src_dir = os.path.split(py)[0]
py = os.path.split(py)[1]
indu = os.path.abspath("./SAT11-INDU")
rand = os.path.abspath("./SAT11-RAND")
runsolver = os.path.abspath("./runsolver")
os.chdir(src_dir)

for scen in [indu, rand]:
    
    with open(scen+"/algorithm_runs.arff") as fp:
        algo_runs = arff.load(fp)
    
    cmd = "%s -C 300 -w /dev/null  python %s --algoruns %s/algorithm_runs.arff" %(runsolver, py, scen)
    print(cmd)
    p = Popen(cmd, shell=True, stdout=PIPE)
    out_, err_ = p.communicate()
    print(out_)
    
    for line in out_.split("\n"):
        if line.startswith("assignment:"):
            line = line.replace("assignment:", "").replace("\n","")
            line = line.replace("u'","'").replace("'", "\"")
            assignment = json.loads(line)
        if line.startswith("permutation:"):
            line = line.replace("permutation:", "").replace("\n","").strip(" ")
            line = line.replace("u'","'").replace("'", "\"")
            permutation = json.loads(line)
            
    get_score(algo_runs, assignment, permutation)
    
    
    
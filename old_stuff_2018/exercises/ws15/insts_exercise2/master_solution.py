__author__="Marius Lindauer"
__version__="1.0"

import sys
import json
import operator
import logging
import random

logging.basicConfig(level=logging.DEBUG)

if len(sys.argv) != 2:
    logging.error("Please provide an CSP instance as argument.")
    sys.exit(1)

#read instance
with open(sys.argv[1]) as fp:
    csp = json.load(fp)

#data
variables = []
domains = {} # variable -> domain range
constraints = []

# define operator mappings
op = {"+" : operator.add,
      "-" : operator.sub,
      "*" : operator.mul,
      "/" : operator.div,
      "%" : operator.mod,
      }

def replace_op(str_):
    return op.get(str_,str_) # replaces str_ by operator if it is in op

#parse Instance
for c in csp:
    if c[0] == "int":
        variables.append(c[1])
        domains[c[1]] = c[2:]
    elif c[0] == "alldifferent":
        for idx_out,term in enumerate(c[1:]):
            if type(term) is list:
                for idx_in,t in enumerate(term):
                    term[idx_in] = replace_op(t)
                c[idx_out+1] = term
            else:
                c[idx_out+1]  = replace_op(term)
        constraints.append(c)
    else:
        logging.error("Not supported: %s" %(c))

logging.debug(variables)
logging.debug(domains)
logging.debug(constraints)

def get_init(domains):
    '''
        returns initial variable assignment / candidate solution
        Args:
            domains: variable name -> range tuple
        Returns:
            variable name -> value
    '''
    assignment = {} # The representation as a dictionary is really inefficient; better would be list
    for v, d in domains.iteritems(): # use iteritems() instead of items() to be more efficient
        assignment[v] = random.randint(d[0],d[1]) #TOOD: we support only integer domains here
    return assignment

def get_neighbor(assignment, variables, domains):
    '''
        returns an neighbor of <assignment> wrt given <domains>
        Args:
            assignment: variable name -> value
            variables: list of variable names
            domains: variable name -> range tuple
        Returns:
            variable name -> value
    '''
    v = random.choice(variables)
    d = domains[v]
    old_value = assignment[v]
    while True: # rejection sampling;  bad style to use "while True"
        new_value = random.randint(d[0],d[1])
        if old_value != new_value:
            assignment[v] = new_value
            return assignment

def is_satisfied(assignment, constraints):
    '''
        checks whether <assignment> satisfies all <constraints>
        Args:
            assignment: variable name -> value
            constraints: list of constraints
        Returns:
            True or Falses
    '''
    for c in constraints:
        if c[0] == "alldifferent": # ineffecient constraint type check
            values = []
            for t in c[1:]:
                if type(t) is list:
                    # replace arithmetic term by value
                    values.append(t[0](assignment.get(t[1],t[1]),
                                       assignment.get(t[2], t[2])))
                else:
                    values.append(assignment.get(t,t)) #replace variable by its values
            if len(values) != len(set(values)):
                return False
        else:
            logging.error("Not support constraint type: %s" %(c[0]))
    return True

#uninformed random walk
assignment = get_init(domains)
steps = 0
while True: #infinite budget
    logging.debug("Current Assignment: %s" %(assignment))
    if is_satisfied(assignment, constraints):
        logging.info("Solution:")
        for v,val in assignment.iteritems():
            print("%s = %d" %(v,val))
        logging.info("Steps: %d" %(steps))
        sys.exit(0)
    assignment = get_neighbor(assignment, variables, domains)
    steps += 1

#!/bin/python

'''
permutation test (one sided p-value)
following the description of: Ensemble-based Prediction of RNA Secondary Structures (Nima Aghaeepour and Holger H. Hoos)

Created on Jul 3, 2014

@author: Marius Lindauer

'''

import sys
import os 
import math
import csv
import argparse
import random
import functools
import logging

class PermutationTester(object):
    '''
       Permutation test on two vectors of performance values
       comparison metric: mean 
    '''

    def __init__(self):
        ''' Constructor '''
        #logging.basicConfig(level=logging.INFO)

    def randomSwap(self, col1,col2):
        '''
            randomly swap elements of the two vectors
        '''
        n = len(col1)
        sc1 = []
        sc2 = []
        for i in range(0,n):
            if (random.random() > 0.5):
                sc1.append(col2[i])
                sc2.append(col1[i])
            else:
                sc1.append(col1[i])
                sc2.append(col2[i])
        avg1 = self.avg(sc1)
        avg2 = self.avg(sc2)
        return avg1-avg2    # betrag?
    
    def getPValue(self, col1,col2,p,avgS):
        '''
            performn p random swaps and return the percentile of the <avgS> in the random swaps averages
        '''
        avgs = [avgS]
        for i in range(0,p-1):
            avgs.append(self.randomSwap(col1,col2))
        logging.debug("AVG Start : " +str(avgS))
        logging.debug("min AVG : "+str(min(avgs)))
        logging.debug("max AVG : "+str(max(avgs)))
        v = self.percentile(avgs,avgS)
        logging.debug("Percentile : "+str(v))
        pValue = v / 100
        return pValue
    
    def percentile(self, N, value):
        ''' returns percentile of value on list N '''
        N.sort()
        #print(N)
        first = N.index(value)
        count = N.count(value)
        #return float(first+count-1)*100/len(N)
        return float(first)*100/len(N)
    
    def avg(self, N):
        return (float(sum(N)) / len(N))
    
    def filter(self,col1,col2,noise):
        '''
            noise filtering based on "Careful Ranking of Multiple Solvers with Timeouts and Ties" A. Van Gelder
        '''
        for i in range(0,len(col1)):
            delta = math.sqrt(noise/2) * math.sqrt((col1[i]+col2[i])/2)  
            avgI = (col1[i]+col2[i])/2
            #print("Delta : "+str(delta))
            #print("AVG : "+str(avgI))
            #print("cols :"+str(col1[i]) + " : "+str(col2[i]))
            if ((col1[i] < col2[i] and col1[i] >= (avgI-delta)) or (col2[i] <= col1[i] and col2[i] >= (avgI-delta))):
                col1[i] = col2[i]
                #print("Set equal")
        return [col1,col2]
    
    def doTest(self, vec1, vec2, alpha=0.05, permutations=10000, name1="", name2="", cutoff=None, par_factor=10, noise=None):
        '''
            perform permutation test
            Args:
                vec1: list of floats (or dictionary)
                vec2: list of floats (or dictionary)
                alpha: significance level
                permutations: number of permutations
                name1, name2: names corresponding to vec1 and vec2
                cutoff, par_factor: if both are set, transform vec1 and vec2 to PAR scores
                noise: noise filtering (float), default deactivated
            Returns:
                True/False: if null hypothesis was rejected (true) or not (false)
                switched: True if alternative hypothesis is vec2 is better than vec1, False: other way around
                pvalue
                
        '''
        if type(vec1) is dict and type(vec2) is dict:
            vec1, vec2 = self.extract_values_in_order(vec1, vec2)
        
        if cutoff and par_factor:
            vec1 = self.convert_to_PARX(vec1, cutoff, par_factor)
            vec2 = self.convert_to_PARX(vec2, cutoff, par_factor)
        
        if vec1 == vec2: # skip at identical vectors
            return False, False, 1
        
        if(noise != None):
            [vec1,vec2] = self.filter(vec1,vec2,float(noise))
        oriAVG = self.avg(vec1) - self.avg(vec2)
        switched = False
        if (oriAVG > 0):
            logging.debug("Alternative Hypothesis: Coloumn %s is better than Coloumn %s " %(name2, name1))
            tmp = vec1
            vec1 = vec2
            vec2 = tmp
            oriAVG *= -1
            switched = True
        else:
            logging.debug("Alternative Hypothesis: Coloumn "+str(name1) + " is better than Coloumn "+str(name2))
        random.seed(1234)
        pValue = self.getPValue(vec1,vec2,permutations,oriAVG)
        if (pValue < alpha):
            logging.debug("Reject Null Hypothesis (p-value: "+str(pValue)+")")
            return True,switched,pValue
        else:
            logging.debug("Don't reject Null Hypothesis (p-value: "+str(pValue)+")")
            return False,switched,pValue

    def convert_to_PARX(self, vec, cutoff, parx):
        '''
            convert values in <vec> to <parx> values wrt to given runtime <cutoff>
            if <parx> < 0, convert to timeout indicator (0, 1) 
        '''
        
        if parx > 0:
            vec_new = map(lambda x: x if x < cutoff else cutoff*parx, vec)
        else:
            vec_new = map(lambda x: 0 if x < cutoff else 1, vec)
        return vec_new
    
    def extract_values_in_order(self, dic1, dic2):
        '''
            extract the values of <dic1> and <dic2> in the same order
            uses the keys of <dic1>
        '''
        vec1 = []
        vec2 = []
        for k in dic1.keys():
            vec1.append(dic1[k])
            vec2.append(dic2[k])
        return vec1, vec2
        
        
#!/bin/python

#===============================================================================
# def test():
#     """Stupid test function"""
#     L = []
#     for i in range(100):
#         L.append(i)
# 
# if __name__ == '__main__':
#     import timeit
#     print(timeit.timeit("test()", setup="from __main__ import test"))
#===============================================================================

import timeit
 
__author__ = "Marius Lindauer"
 
def string_con1():
    '''
        inefficient in old python versions
    '''
    list = range(100)
    s = ""
    for substring in list:
        s += str(substring)
         
def string_con2():
    list = range(100)
    s = "".join(str(x) for x in list)
    
def string_con3():
    list = range(100)
    s = "".join([str(x) for x in list])    
     
def string_con4():
    list = range(100)
    s = "".join(map(str, list))
    
def loop1():
    list = []
    for n in range(100):
        list.append(str(n))

def loop2():
    list = map(str, range(100))
    
def dict1():
    '''
        not an issue in Python 3.4
    '''
    dict_ = dict((str(x),x) for x in range(100000))
    list = []
    for key, value in dict_.items():
        pass
     
def dict2():
    dict_ = dict((str(x),x) for x in range(100000))
    list = []
    for key, value in dict.iteritems():
        pass
    
def range1():
    '''
        not an issue in Python 3.4
    '''
    for i in range(1,100000):
        pass  

def range2():
    for i in xrange(1,100000):
        pass  

if __name__ == '__main__':
     
    #===========================================================================
    # print(timeit.timeit("string_con1()", setup="from __main__ import string_con1"))
    # print(timeit.timeit("string_con2()", setup="from __main__ import string_con2"))
    # print(timeit.timeit("string_con3()", setup="from __main__ import string_con3"))
    # print(timeit.timeit("string_con4()", setup="from __main__ import string_con4"))
    #===========================================================================
    
    #===========================================================================
    # print(timeit.timeit("loop1()", setup="from __main__ import loop1"))
    # print(timeit.timeit("loop2()", setup="from __main__ import loop2"))
    #===========================================================================
    
    #===========================================================================
    # print(timeit.timeit("dict1()", setup="from __main__ import dict1", number=1000))
    # print(timeit.timeit("dict1()", setup="from __main__ import dict1", number=1000))
    #===========================================================================

    #===========================================================================
    # print(timeit.timeit("range1()", setup="from __main__ import range1", number=1000))
    # print(timeit.timeit("range2()", setup="from __main__ import range2", number=1000))
    #===========================================================================
    
    print(timeit.timeit("gen1()", setup="from __main__ import gen1", number=100))
    print(timeit.timeit("gen1()", setup="from __main__ import gen1", number=100))
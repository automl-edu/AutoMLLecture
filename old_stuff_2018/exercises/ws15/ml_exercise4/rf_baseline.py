#!/bin/python

import sys
import numpy
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


for idx in range(1,6):

    X_train = numpy.loadtxt("set%d/x_train.np" %(idx))  
    y_train = numpy.loadtxt("set%d/y_train.np" %(idx))  
    X_test = numpy.loadtxt("set%d/x_test.np" %(idx))
    y_test = numpy.loadtxt("set%d/y_test.np" %(idx))  
    
    
    rf = RandomForestClassifier(n_estimators=10)
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy: %.4f" %(acc))
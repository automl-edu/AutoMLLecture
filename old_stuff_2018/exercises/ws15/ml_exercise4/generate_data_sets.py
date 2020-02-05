#!/bin/python

from sklearn.datasets import make_classification
import numpy
import os


def write_sets(X,y, name):
    
    test_fraction = 0.1
    n = X.shape[0]
    n_train = int(n * (1 - 0.1))
    
    X_train = X[:n_train+1, :]
    y_train = y[:n_train+1]
    
    X_test = X[n_train+1:, :]
    y_test = y[n_train+1:]
    
    try:
        os.mkdir(name)
    except OSError:
        pass
    
    numpy.savetxt(os.path.join(name, "x_train.np"), X_train)
    numpy.savetxt(os.path.join(name, "y_train.np"), y_train, "%d")

    numpy.savetxt(os.path.join(name, "x_test.np"), X_test)
    numpy.savetxt(os.path.join(name, "y_test.np"), y_test, "%d")
    

X, y = make_classification(n_samples=1000, 
                           n_features=6, 
                           n_informative=6, 
                           n_redundant=0, 
                           n_repeated=0, 
                           n_classes=2,
                           n_clusters_per_class=3)
write_sets(X, y, "set1")


X, y = make_classification(n_samples=2000, 
                           n_features=10, 
                           n_informative=6, 
                           n_redundant=4, 
                           n_repeated=0, 
                           n_classes=2,
                           n_clusters_per_class=2)

write_sets(X, y, "set2")

X, y = make_classification(n_samples=4000, 
                           n_features=10, 
                           n_informative=2, 
                           n_redundant=6, 
                           n_repeated=2, 
                           n_classes=2,
                           n_clusters_per_class=2)

write_sets(X, y, "set3")

X, y = make_classification(n_samples=6000, 
                           n_features=12, 
                           n_informative=12, 
                           n_redundant=0, 
                           n_repeated=0, 
                           n_classes=2,
                           n_clusters_per_class=4)

write_sets(X, y, "set4")

X, y = make_classification(n_samples=1000, 
                           n_features=100, 
                           n_informative=10, 
                           n_redundant=90, 
                           n_repeated=0, 
                           n_classes=2,
                           n_clusters_per_class=4)

write_sets(X, y, "set5")




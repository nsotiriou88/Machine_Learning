#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 12:14:06 2019

@author: Nicholas
"""

# Load Libraries
import pandas as pd
import sklearn 
from sklearn.model_selection import train_test_split
import sklearn.ensemble
import numpy as np
import lime
import lime.lime_tabular
import shap
import xgboost as xgb
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
import seaborn as sns
import time
#%matplotlib inline
 
# Load Boston Housing Data
X,y = shap.datasets.boston()
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# K Nearest Neighbor
knn = sklearn.neighbors.KNeighborsRegressor()
knn.fit(X_train, y_train)
 
# Create the SHAP Explainers
# SHAP has the following explainers: deep, gradient, kernel, linear, tree, sampling
# Must use Kernel method on knn
# Summarizing the data with k-Means is a trick to speed up the processing 
"""
Rather than use the whole training set to estimate expected values, we summarize with
a set of weighted kmeans, each weighted by the number of points they represent.
Running without kmeans took 1 hr 6 mins 7 sec. 
Running with kmeans took 2 min 47 sec.
Boston Housing is a small dataset.
Running SHAP on models that require the Kernel method becomes prohibitive. 
""" 
# build the kmeans summary
X_train_summary = shap.kmeans(X_train, 10)
 
# using the kmeans summary
t0 = time.time()
explainerKNN = shap.KernelExplainer(knn.predict,X_train_summary)
shap_values_KNN_test = explainerKNN.shap_values(X_test)
t1 = time.time()
timeit=t1-t0
timeit
 
# without kmeans
# a test run took 3967.6232330799103 seconds
"""
t0 = time.time()
explainerKNN = shap.KernelExplainer(knn.predict, X_train)
shap_values_KNN_test = explainerKNN.shap_values(X_test)

t1 = time.time()
timeit=t1-t0
timeit
"""
# now we can plot the SHAP explainer
shap.force_plot(explainerKNN.expected_value, shap_values_KNN_test[j], X_test.iloc[[j]])

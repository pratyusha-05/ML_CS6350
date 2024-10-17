#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np

trainDf = pd.read_csv('Datasets/concrete/train.csv')
testDf = pd.read_csv('Datasets/concrete/test.csv')

X_train = trainDf.iloc[:, :-1].values 
y_train = trainDf.iloc[:, -1].values 
X_test = testDf.iloc[:, :-1].values
y_test = testDf.iloc[:, -1].values

X_train = np.concatenate([np.ones((X_train.shape[0], 1)), X_train], axis=1)
X_test = np.concatenate([np.ones((X_test.shape[0], 1)), X_test], axis=1)

optimal_weight_vector = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train

print("Optimal Weight Vector:", optimal_weight_vector)


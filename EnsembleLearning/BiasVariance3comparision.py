#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier

columns = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']

trainDf = pd.read_csv("Datasets/bank/train.csv", names=columns)
testDf = pd.read_csv("Datasets/bank/test.csv", names=columns)

X_train = pd.get_dummies(trainDf.drop('y', axis=1))
y_train = trainDf['y'].apply(lambda x: 1 if x == 'yes' else 0).values  

X_test = pd.get_dummies(testDf.drop('y', axis=1))
y_test = testDf['y'].apply(lambda x: 1 if x == 'yes' else 0).values

X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

num_experiments = 100
num_trees = 100

def compute_bias_variance(predictions, y_true):
    avg_pred = np.mean(predictions, axis=0)
    bias = np.mean((avg_pred - y_true) ** 2)
    variance = np.mean(np.var(predictions, axis=0))
    squared_error = bias + variance

    return bias, variance, squared_error

single_tree_preds = []
bagged_trees_preds = []
random_forest_preds = []

for _ in range(num_experiments):
    sample_indices = np.random.choice(len(X_train), size=1000, replace=False)
    X_sample, y_sample = X_train.iloc[sample_indices], y_train[sample_indices]

    single_tree = DecisionTreeClassifier()
    single_tree.fit(X_sample, y_sample)
    single_tree_preds.append(single_tree.predict(X_test))

    bagging_model = BaggingClassifier(DecisionTreeClassifier(), n_estimators=num_trees)
    bagging_model.fit(X_sample, y_sample)
    bagged_trees_preds.append(bagging_model.predict(X_test))

    random_forest = RandomForestClassifier(n_estimators=num_trees)
    random_forest.fit(X_sample, y_sample)
    random_forest_preds.append(random_forest.predict(X_test))

single_tree_preds = np.array(single_tree_preds)
bagged_trees_preds = np.array(bagged_trees_preds)
random_forest_preds = np.array(random_forest_preds)

print(f"Single Tree - Bias: {bias_single}, Variance: {variance_single}, Squared Error: {error_single}")
print(f"Bagged Trees - Bias: {bias_bagged}, Variance: {variance_bagged}, Squared Error: {error_bagged}")
print(f"Random Forest - Bias: {bias_rf}, Variance: {variance_rf}, Squared Error: {error_rf}")


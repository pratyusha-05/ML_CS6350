#!/usr/bin/env python
# coding: utf-8

# In[26]:


import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from collections import Counter
from sklearn.utils import resample
import matplotlib.pyplot as plt

class TreeNode:
    def __init__(self, attribute, attribute_name, is_leaf, label, depth):
        self.attribute = attribute
        self.attribute_name = attribute_name
        self.children = {}
        self.is_leaf = is_leaf
        self.label = label
        self.depth = depth

    def add_child(self, child_node, attr_value):
        self.children[attr_value] = child_node

    def predict(self, x):
        if self.is_leaf:
            return self.label
        value = x[self.attribute]
        child_node = self.children.get(value)
        if child_node is not None:
            return child_node.predict(x)
        else:
            return self.label
    
class DecisionTreeClassifier:
    def __init__(self, max_depth=np.inf):
        self.root = None
        self.max_depth = max(1, max_depth)

    def fit(self, x, y):
        feature_names = list(range(x.shape[1]))
        feature_list = np.arange(x.shape[1])
        self.root = self._build_tree(x, y, feature_names, feature_list)

    def _build_tree(self, x, y, feature_names, features, depth=0):
        if depth >= self.max_depth or len(features) == 0 or len(np.unique(y)) == 1:
            return TreeNode(None, None, True, self._majority_class(y), depth)

        best_feature = self._find_best_split(x, y, features)
        
        root = TreeNode(best_feature, feature_names[best_feature], False, 
                        self._majority_class(y), depth)

        remaining_features = [f for f in features if f != best_feature]
        for value in np.unique(x[:, best_feature]):
            mask = x[:, best_feature] == value
            if not np.any(mask):
                root.add_child(TreeNode(None, None, True, self._majority_class(y), depth + 1), value)
            else:
                root.add_child(self._build_tree(x[mask], y[mask], feature_names, remaining_features, depth + 1), value)

        return root

    def _majority_class(self, y):
        return Counter(y).most_common(1)[0][0]

    def _find_best_split(self, x, y, features):
        best_gain = -float('inf')
        best_feature = None

        for feature in features:
            gain = self._information_gain(x, y, feature)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
        return best_feature

    def _information_gain(self, x, y, feature):
        x_feature = x[:, feature]
        entropy_before = self._entropy(y)
        entropy_after = sum((x_feature == value).mean() * self._entropy(y[x_feature == value])
                            for value in np.unique(x_feature))
        return entropy_before - entropy_after

    def _entropy(self, y):
        if y.dtype.kind not in 'iu': 
            unique_classes, y_integer = np.unique(y, return_inverse=True)
        else:
            y_integer = y
        
        probabilities = np.bincount(y_integer) / len(y_integer)
        return -np.sum(probabilities * np.log2(probabilities + 1e-10))

    def predict(self, x):
        return [self.root.predict(a) for a in x]
    
class BaggedTrees:
    def __init__(self, num_trees):
        self.num_trees = num_trees
        self.trees = []

    def fit(self, x, y):
        n_samples = x.shape[0]
        for _ in range(self.num_trees):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            x_resampled, y_resampled = x[indices], y[indices]
            dt = DecisionTreeClassifier(max_depth = np.inf)
            dt.fit(x_resampled, y_resampled)
            self.trees.append(dt)

    def predict(self, x):
        all_predictions = np.array([tree.predict(x) for tree in self.trees])
        aggregated_predictions = np.sum(all_predictions, axis=0)
        return np.sign(aggregated_predictions)

    
def evaluate_bagged_trees_model(n_trees, x_train, y_train, x_test, y_test):
    num_trees = n_trees
    train_errors = []
    test_errors = []

    for n in range(1, num_trees+1):
        print(n)
        bagged_trees_classifier = BaggedTrees(n) 
        bagged_trees_classifier.fit(x_train, y_train)

        train_pred = bagged_trees_classifier.predict(x_train)
        test_pred = bagged_trees_classifier.predict(x_test)

        train_error = 1 - accuracy_score(y_train, train_pred)
        test_error = 1 - accuracy_score(y_test, test_pred)

        train_errors.append(train_error)
        test_errors.append(test_error)

    plt.figure(figsize=(12, 5))
    plt.plot(range(1, num_trees+1), train_errors, label='Train Error')
    plt.plot(range(1, num_trees+1), test_errors, label='Test Error')
    plt.xlabel('Number of Trees')
    plt.ylabel('Error')
    plt.title('Training and Test Errors vs. Number of Trees')
    plt.legend()
    plt.show()

def main():
    columns = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']
    trainDf = pd.read_csv("Datasets/bank/train.csv", names=columns)
    x_train = trainDf.drop('y', axis=1).values
    y_train = trainDf['y'].apply(lambda x: 1 if x == 'yes' else 0).values.astype(float)

    testDf = pd.read_csv("Datasets/bank/test.csv", names=columns)
    x_test = testDf.drop('y', axis=1).values
    y_test = testDf['y'].apply(lambda x: 1 if x == 'yes' else 0).values.astype(float)
    
    n_trees = 500
    evaluate_bagged_trees_model(n_trees, x_train, y_train, x_test, y_test)

main()


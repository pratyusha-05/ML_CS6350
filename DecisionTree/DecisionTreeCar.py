#!/usr/bin/env python
# coding: utf-8

# In[397]:


import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from collections import Counter

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


# In[398]:


class DecisionTreeClassifier:
    def __init__(self, heuristic, max_depth=np.inf):
        self.root = None
        self.max_depth = max(1, max_depth)
        self.heuristic = heuristic
        self.longest_path = 0

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
            gain = self._calculate_gain(x, y, feature)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
        return best_feature

    def _calculate_gain(self, x, y, feature):
        if self.heuristic == 'information_gain':
            return self._information_gain(x, y, feature)
        elif self.heuristic == 'majority_error':
            return self._majority_error(x, y, feature)
        elif self.heuristic == 'gini_index':
            return self._gini(x, y, feature)

    def _information_gain(self, x, y, feature):
        x_feature = x[:, feature]
        entropy_before = self._entropy(y)
        entropy_after = sum((x_feature == value).mean() * self._entropy(y[x_feature == value])
                            for value in np.unique(x_feature))
        return entropy_before - entropy_after


    def _majority_error(self, x, y, feature):
        total_samples = len(y)
        majority_class_count = Counter(y).most_common(1)[0][1]
        error_before = 1 - (majority_class_count / total_samples)
    
        error_after = 0
        for X in np.unique(x[:, feature]):
            subset_mask = (x[:, feature] == X)
            subset_y = y[subset_mask]
            if len(subset_y) == 0:
                continue
            subset_majority_count = Counter(subset_y).most_common(1)[0][1]
            subset_error = 1 - (subset_majority_count / len(subset_y))
            error_after += (len(subset_mask) / total_samples) * subset_error
        return error_before - error_after
    

    def _gini(self, x, y, feature):
        values = np.unique(x[:, feature])
        
        gini_before = 1
        for c in np.unique(y):
            gini_before -= (np.mean(y == c)) ** 2
            
        gini_after = 0
        for value in values:
            subset_of_x = (x[:, feature] == value)
            subset_of_y = y[subset_of_x]
            gini_of_subset = 1 - sum(((np.mean(subset_of_y == c)) ** 2 for c in np.unique(subset_of_y)))
            gini_after += subset_of_x.mean() * gini_of_subset                
        return gini_before - gini_after

    def _entropy(self, y):
        if y.dtype.kind not in 'iu': 
            unique_classes, y_integer = np.unique(y, return_inverse=True)
        else:
            y_integer = y
        
        probabilities = np.bincount(y_integer) / len(y_integer)
        return -np.sum(probabilities * np.log2(probabilities + 1e-10))

    def predict(self, x):
        return [self.root.predict(a) for a in x]
      
    # call this method to see the decision tree structure
    def print_tree(self):
        self._print_node(self.root)

    def _print_node(self, node, prefix=""):
        if node.is_leaf:
            print(f"{prefix}Leaf: {node.label}")
        else:
            print(f"{prefix}Split on feature {node.attribute_name}")
            for value, child in node.children.items():
                print(f"{prefix}├── value = {value}:")
                self._print_node(child, prefix + "│   ")


# In[401]:


def evaluate_model(model, x_train, y_train, x_test, y_test):
    y_train_pred = model.predict(x_train.values)
    y_test_pred = model.predict(x_test.values)
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    return 1 - train_acc, 1 - test_acc

def main():
    columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'label']
    trainDf = pd.read_csv("Datasets/car/train.csv", names = columns)
    x_train = trainDf.drop('label', axis=1)
    y_train = trainDf['label']

    testDf = pd.read_csv("Datasets/car/test.csv",names = columns)
    x_test = testDf.drop('label', axis=1)
    y_test = testDf['label']
    
    results = {}
    
    for depth in range(1, 7): 
        results[depth] = []
        for heuristic in ['information_gain', 'gini_index', 'majority_error']:
            model = DecisionTreeClassifier(heuristic=heuristic, max_depth=depth)
            model.fit(x_train.values, y_train.values)
            train_error, test_error = evaluate_model(model, x_train, y_train, x_test, y_test)
            if depth not in results:
                results[depth] = [[heuristic, round(train_error, 3), round(test_error, 3)]]
            else:
                results[depth].append([heuristic, round(train_error, 3), round(test_error, 3)])

    print('       Information Gain\t Gini Index \t Majority Error')
    print("Depth \t Train \t Test \t Train \t Test \t Train \t Test")
    for depth, errors in results.items():
        print(' ', depth, end='\t')
        print(errors[0][1], '\t', errors[0][2], '\t', errors[1][1], '\t', errors[1][2], '\t', errors[2][1], '\t', errors[2][2])

main()


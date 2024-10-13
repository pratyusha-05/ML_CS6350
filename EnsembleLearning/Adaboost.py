#!/usr/bin/env python
# coding: utf-8

# In[29]:


import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

class AdaBoost:
    def __init__(self, T):
        self.T = T
        self.learning_rate = 0.5
        self.learner_weights = []
        self.stumps = []
        self.training_errors = []  
        self.test_errors = []  
        self.stump_errors = []  
        self.attributes = []
        self.thresholds = []
        self.predictions_below_threshold = []
        self.predictions_above_threshold = []

    def numeric_conversion(self, X):
        df = pd.DataFrame(X)
        df = df.apply(lambda col: pd.to_numeric(col, errors='coerce').fillna(0))
        return df.to_numpy()
    
    def fit(self, X, y):
        y = y.astype(float)
        n = len(y)
        weights = np.ones(n) / n
        X = self.numeric_conversion(X)

        for t in range(self.T):
            attribute, threshold, prediction_below_threshold, prediction_above_threshold = self.train_stump(X, y, weights)
            self.attributes.append(attribute)
            self.thresholds.append(threshold)
            self.predictions_below_threshold.append(prediction_below_threshold)
            self.predictions_above_threshold.append(prediction_above_threshold)

            predictions = np.where(X[:, attribute] <= threshold, prediction_below_threshold, prediction_above_threshold).astype(float)
            weighted_error = np.sum(weights * (predictions != y))

            learner_weight = self.learning_rate * np.log((1 - weighted_error) / max(weighted_error, 1e-10)) if weighted_error > 0 else 1.0
            self.learner_weights.append(learner_weight)

            weights *= np.exp(-learner_weight * y * predictions)
            weights /= np.sum(weights)

            train_predictions = self.predict(X_train)
            test_predictions = self.predict(X_test)
            train_error = 1 - accuracy_score(y_train, train_predictions)
            test_error = 1 - accuracy_score(y_test, test_predictions)
        
            self.training_errors.append(train_error)
            self.test_errors.append(test_error)
            self.stump_errors.append(weighted_error / n)

    def train_stump(self, X, y, weights):
        optimal_error = float('inf')
        optimal_attribute = None
        optimal_threshold = None
        optimal_prediction_below_threshold = None
        optimal_prediction_above_threshold = None

        attribute, threshold = self.find_best_split(X, y, weights)
        
        for i in [-1, 1]:
            predictions = np.where(X[:, attribute] <= threshold, i, -i)
            error = np.sum(weights * (predictions != y))

            if error < optimal_error:
                optimal_error = error
                optimal_prediction_below_threshold = i
                optimal_prediction_above_threshold = -i
                optimal_attribute = attribute
                optimal_threshold = threshold

        return optimal_attribute, optimal_threshold, optimal_prediction_below_threshold, optimal_prediction_above_threshold

    def calculate_information_gain(self, X, y, attribute, threshold, weights):
        n = len(y)
        below_threshold = X[:, attribute] <= threshold
        above_threshold = X[:, attribute] > threshold

        below_weight = np.sum(weights[below_threshold])
        above_weight = np.sum(weights[above_threshold])

        if below_weight == 0 or above_weight == 0:
            return 0

        below_entropy = -np.sum(weights[below_threshold] * np.log2(weights[below_threshold] / below_weight))
        above_entropy = -np.sum(weights[above_threshold] * np.log2(weights[above_threshold] / above_weight))

        total_entropy = (below_weight / n) * below_entropy + (above_weight / n) * above_entropy
        return total_entropy

    def find_best_split(self, X, y, weights):
        num_features = X.shape[1]
        optimal_threshold = 0
        optimal_attribute = 0
        min_entropy = float('inf')

        for attribute in range(num_features):
            attribute_values = np.unique(X[:, attribute])
            thresholds = (attribute_values[:-1] + attribute_values[1:]) / 2

            for threshold in thresholds:
                entropy = self.calculate_information_gain(X, y, attribute, threshold, weights)

                if entropy < min_entropy:
                    min_entropy = entropy
                    optimal_threshold = threshold
                    optimal_attribute = attribute

        return optimal_attribute, optimal_threshold

    def predict(self, X):
        n = X.shape[0]
        predictions = np.zeros(n)

        for t in range(len(self.learner_weights)):
            learner_weight = self.learner_weights[t]
            attribute = self.attributes[t]
            threshold = self.thresholds[t]
            prediction_below_threshold = self.predictions_below_threshold[t]
            prediction_above_threshold = self.predictions_above_threshold[t]

            stump_predictions = np.where(X[:, attribute] <= threshold, prediction_below_threshold, prediction_above_threshold)
            predictions += learner_weight * stump_predictions

        return np.sign(predictions)


# In[33]:


columns = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']
train_df = pd.read_csv("Datasets/bank/train.csv", names=columns)
X_train = train_df.drop('y', axis=1).values
y_train = train_df['y'].apply(lambda x: 1 if x == 'yes' else -1).values

test_df = pd.read_csv("Datasets/bank/test.csv", names=columns)
X_test = test_df.drop('y', axis=1).values
y_test = test_df['y'].apply(lambda x: 1 if x == 'yes' else -1).values
    
T = 500 
adaboost = AdaBoost(T)
adaboost.fit(X_train, y_train)

def evaluate_model():
    train_predictions = adaboost.predict(X_train)
    test_predictions = adaboost.predict(X_test)
    train_error = 1 - accuracy_score(y_train, train_predictions)
    test_error = 1 - accuracy_score(y_test, test_predictions)

    print("Train Error:", train_error)
    print("Test Error:", test_error)
    display_plot()

def display_plot():
    plt.figure(figsize=(15, 7))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, T + 1), adaboost.training_errors, label='Training Error', marker='o')
    plt.plot(range(1, T + 1), adaboost.test_errors, label='Test Error', marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.title('Training and Test Errors vs. Iteration')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, T + 1), adaboost.stump_errors, label='Decision Stump Error', marker='o')
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.title('Decision Stump Errors vs. Iteration')
    plt.legend()

    plt.tight_layout()
    plt.show()
    
evaluate_model()
    


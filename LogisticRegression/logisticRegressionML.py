#ml

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_gradient(features, labels, weights):
    num_samples = features.shape[0]
    predictions = sigmoid(np.dot(features, weights))
    errors = predictions - labels

    gradient = np.dot(features.T, errors) / num_samples

    return gradient

def optimize_with_sgd(x_train, y_train, x_test, y_test, learning_rate, decay, epochs):
    num_samples, num_features = x_train.shape
    weights = np.zeros(num_features)  
    training_errors = []
    test_errors = []
    objective_values = []

    for epoch in range(epochs):
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        x_train = x_train[indices]
        y_train = y_train[indices]

        for step in range(1, num_samples + 1):
            current_lr = learning_rate / (1 + (learning_rate / decay) * step)
            idx = step - 1
            grad = compute_gradient(x_train[idx:idx+1], y_train[idx:idx+1], weights)
            weights -= current_lr * grad

        train_predictions = sigmoid(np.dot(x_train, weights)) > 0.5
        test_predictions = sigmoid(np.dot(x_test, weights)) > 0.5

        train_error = np.mean(train_predictions != y_train) * 100
        test_error = np.mean(test_predictions != y_test) * 100

        likelihood = np.mean(y_train * np.log(sigmoid(np.dot(x_train, weights))) + (1 - y_train) * np.log(1 - sigmoid(np.dot(x_train, weights))))
        objective_value = -likelihood

        training_errors.append(train_error)
        test_errors.append(test_error)
        objective_values.append(objective_value)

    return weights, training_errors, test_errors, objective_values


train_df = pd.read_csv("Datasets/bank-note/train.csv", header=None)
test_df = pd.read_csv("Datasets/bank-note/test.csv", header=None)

train_features = train_df.iloc[:, :-1].values
train_labels = train_df.iloc[:, -1].values
test_features = test_df.iloc[:, :-1].values
test_labels = test_df.iloc[:, -1].values

variances = [0.1, 0.5, 1, 3, 5, 10, 100]
gamma_0 = 0.1
d = 0.01
max_epochs = 100

for var in variances:
    print(f"Variance: {var}")
    params, train_error_list, test_error_list, objective_value_list = sgd_optimizer(
        train_features, train_labels, test_features, test_labels, var, gamma_0, d, max_epochs
    )
    print(f"Final Training Error: {train_error_list[-1]:.2f}%, Final Test Error: {test_error_list[-1]:.2f}%")


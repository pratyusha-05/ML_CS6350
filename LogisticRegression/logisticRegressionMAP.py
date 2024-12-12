import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def sigmoid_function(z):
    return 1 / (1 + np.exp(-z))

def calculate_gradient(features, labels, params, variance):
    num_samples = features.shape[0]
    predictions = sigmoid_function(np.dot(features, params))
    errors = predictions - labels

    likelihood_gradient = np.dot(features.T, errors) / num_samples

    prior_gradient = params / variance

    return likelihood_gradient + prior_gradient

def sgd_optimizer(train_features, train_labels, test_features, test_labels, variance, gamma_0, d, max_epochs):
    num_samples, num_features = train_features.shape
    params = np.zeros(num_features)  
    training_error_list = []
    test_error_list = []
    objective_value_list = []

    for epoch in range(max_epochs):
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        train_features = train_features[indices]
        train_labels = train_labels[indices]

        for step in range(1, num_samples + 1):
            learning_rate = gamma_0 / (1 + (gamma_0 / d) * step)
            idx = step - 1
            gradient = calculate_gradient(train_features[idx:idx+1], train_labels[idx:idx+1], params, variance)
            params -= learning_rate * gradient

        train_preds = sigmoid_function(np.dot(train_features, params)) > 0.5
        test_preds = sigmoid_function(np.dot(test_features, params)) > 0.5

        train_error = np.mean(train_preds != train_labels) * 100
        test_error = np.mean(test_preds != test_labels) * 100

        likelihood = np.mean(train_labels * np.log(sigmoid_function(np.dot(train_features, params))) + (1 - train_labels) * np.log(1 - sigmoid_function(np.dot(train_features, params))))
        prior = -np.sum(params ** 2) / (2 * variance)
        objective_value = -likelihood - prior

        training_error_list.append(train_error)
        test_error_list.append(test_error)
        objective_value_list.append(objective_value)

    return params, training_error_list, test_error_list, objective_value_list

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

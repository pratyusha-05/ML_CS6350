import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from scipy.optimize import minimize

train_data = pd.read_csv("Datasets/bank-note/train.csv", header=None)
test_data = pd.read_csv("Datasets/bank-note/test.csv", header=None)

train_data.iloc[:, -1] = train_data.iloc[:, -1].replace({1: 1, 0: -1})
test_data.iloc[:, -1] = test_data.iloc[:, -1].replace({1: 1, 0: -1})

# Primal SVM: Stochastic Sub-Gradient Descent
def primal_svm_sgd(X, y, C, initial_learning_rate, decay_factor, epochs):
    num_samples, num_features = X.shape
    weights = np.zeros(num_features)  
    bias = 0  
    update_count = 0
    
    for epoch in range(epochs):
        X, y = shuffle(X, y, random_state=epoch)
        
        for i in range(num_samples):
            update_count += 1
            learning_rate = initial_learning_rate / (1 + (initial_learning_rate / decay_factor) * update_count)
            margin = y[i] * (np.dot(X[i], weights) + bias)
            
            if margin < 1:
                weights = (1 - learning_rate) * weights + learning_rate * C * y[i] * X[i]
                bias += learning_rate * C * y[i]
            else:
                weights = (1 - learning_rate) * weights

    return weights, bias

def dual_svm_objective(alpha, X, y):
    gram_matrix = np.dot(X, X.T) * np.outer(y, y)
    return 0.5 * np.dot(alpha, np.dot(gram_matrix, alpha)) - np.sum(alpha)

def dual_constraint(alpha, y):
    return np.dot(alpha, y)

def train_dual_svm(X, y, C):
    num_samples = len(y)
    alpha_bounds = [(0, C) for _ in range(num_samples)]
    constraint = {'type': 'eq', 'fun': dual_constraint, 'args': (y,)}

    initial_alpha = np.zeros(num_samples)
    result = minimize(dual_svm_objective, initial_alpha, args=(X, y), method='SLSQP', bounds=alpha_bounds, constraints=constraint)
    optimal_alpha = result.x

    weights = np.dot(optimal_alpha * y, X)
    support_vector_indices = (optimal_alpha > 1e-5) & (optimal_alpha < C)
    bias = np.mean(y[support_vector_indices] - np.dot(X[support_vector_indices], weights))
    return weights, bias

C_values = [100 / 873, 500 / 873, 700 / 873]
primal_results = []
dual_results = []

X_train, y_train = train_data.iloc[:, :-1].values, train_data.iloc[:, -1].values
X_test, y_test = test_data.iloc[:, :-1].values, test_data.iloc[:, -1].values

for C in C_values:
    w_primal, b_primal = primal_svm_sgd(X_train, y_train, C, initial_learning_rate=0.1, decay_factor=0.01, epochs=100)
    train_error_primal = np.mean(np.sign(np.dot(X_train, w_primal) + b_primal) != y_train)
    test_error_primal = np.mean(np.sign(np.dot(X_test, w_primal) + b_primal) != y_test)
    
    primal_results.append({'C': C, 'weights': w_primal, 'bias': b_primal, 'train_error': train_error_primal, 'test_error': test_error_primal})

    w_dual, b_dual = train_dual_svm(X_train, y_train, C)
    train_error_dual = np.mean(np.sign(np.dot(X_train, w_dual) + b_dual) != y_train)
    test_error_dual = np.mean(np.sign(np.dot(X_test, w_dual) + b_dual) != y_test)
    
    dual_results.append({'C': C, 'weights': w_dual, 'bias': b_dual, 'train_error': train_error_dual, 'test_error': test_error_dual})

for primal, dual in zip(primal_results, dual_results):
    C = primal['C']
    weight_diff = np.linalg.norm(primal['weights'] - dual['weights'])
    bias_diff = np.abs(primal['bias'] - dual['bias'])
    train_error_diff = np.abs(primal['train_error'] - dual['train_error'])
    test_error_diff = np.abs(primal['test_error'] - dual['test_error'])

    print(f"C: {C}")
    print("Primal SVM:")
    print(f"  Weights: {primal['weights']}")
    print(f"  Bias: {primal['bias']}")
    print(f"  Training Error: {primal['train_error']}")
    print(f"  Testing Error: {primal['test_error']}")

    print("Dual SVM:")
    print(f"  Weights: {dual['weights']}")
    print(f"  Bias: {dual['bias']}")
    print(f"  Training Error: {dual['train_error']}")
    print(f"  Testing Error: {dual['test_error']}")

    print("Differences:")
    print(f"  Weight Difference: {weight_diff}")
    print(f"  Bias Difference: {bias_diff}")
    print(f"  Training Error Difference: {train_error_diff}")
    print(f"  Testing Error Difference: {test_error_diff}")
    print()

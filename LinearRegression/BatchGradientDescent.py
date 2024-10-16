#!/usr/bin/env python
# coding: utf-8

# In[16]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def compute_cost(X, y, theta):
    m = len(y)
    h = X.dot(theta)
    J = np.mean((h - y) ** 2) / 2
    return J

def gradient_descent(X, y, theta, alpha, epsilon, max_iterations=100000):
    m = len(y)
    cost_history = []
    
    for iteration in range(max_iterations):
        h = X.dot(theta)
        gradient = X.T.dot(h - y) / m
        
        theta_new = theta - np.clip(alpha * gradient, -1e150, 1e150)
        
        if np.linalg.norm(theta_new - theta) < epsilon:
            print(f"Convergence achieved after {iteration + 1} iterations")
            break
        
        theta = theta_new
        cost_history.append(compute_cost(X, y, theta))
    
    return theta, cost_history

def load_data(filepath):
    data = pd.read_csv(filepath)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    X = np.c_[np.ones(X.shape[0]), X]
    X[:, 1:] = (X[:, 1:] - np.mean(X[:, 1:], axis=0)) / np.std(X[:, 1:], axis=0)
    
    return X, y

def tune_learning_rate(X, y, initial_theta, learning_rates, epsilon):
    for alpha in learning_rates:
        print(f"Testing learning rate: {alpha}")
        theta, cost_history = gradient_descent(X, y, initial_theta, alpha, epsilon)
        if len(cost_history) < 100000 and all(np.isfinite(cost_history)):
            return alpha, theta, cost_history
    return None, None, None

def plot_cost_history(cost_history, alpha):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cost_history) + 1), cost_history)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title(f'Cost vs Iterations (Learning Rate = {alpha})')
    plt.yscale('log') 
    ax = plt.gca()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.2f}"))
    plt.show()

def main():
    X_train, y_train = load_data('Datasets/concrete/train.csv')
    X_test, y_test = load_data('Datasets/concrete/test.csv')
    
    initial_theta = np.zeros(X_train.shape[1])
    epsilon = 1e-6
    
    learning_rates = [1.0, 0.5, 0.25, 0.125]
    best_alpha, optimal_theta, cost_history = tune_learning_rate(X_train, y_train, initial_theta, learning_rates, epsilon)
    
    if best_alpha is None:
        print("Failed to find a suitable learning rate")
        return
    
    plot_cost_history(cost_history, best_alpha)
    test_cost = compute_cost(X_test, y_test, optimal_theta)
    
    print('Optimized parameter values:', optimal_theta)
    print('Chosen learning rate:', best_alpha)
    print('Test set cost:', test_cost)

main()


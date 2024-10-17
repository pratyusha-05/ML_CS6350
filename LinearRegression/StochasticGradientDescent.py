#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

def compute_cost(X, y, theta):
    m = len(y)
    h = X.dot(theta)
    J = np.mean((h - y) ** 2) / 2
    return J

def stochastic_gradient_descent(X, y, initial_theta, alpha, num_epochs, batch_size=1):
    m, n = X.shape
    theta = initial_theta.copy()
    cost_history = []

    for epoch in range(num_epochs):
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        for i in range(0, m, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]

            h = X_batch.dot(theta)
            gradient = X_batch.T.dot(h - y_batch) / batch_size
            theta = theta - alpha * gradient

        cost = compute_cost(X, y, theta)
        cost_history.append(cost)

    return theta, cost_history

def load_data(filepath):
    data = pd.read_csv(filepath)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    X = np.c_[np.ones(X.shape[0]), X]
    
    X[:, 1:] = (X[:, 1:] - np.mean(X[:, 1:], axis=0)) / np.std(X[:, 1:], axis=0)
    
    return X, y

def plot_cost_history(cost_history, alpha, num_points=100):
    total_points = len(cost_history)
    
    indices = np.linspace(0, total_points - 1, num_points, dtype=int)
    x = np.array([i for i in indices])
    y = np.array([cost_history[i] for i in indices])
    
    x_smooth = np.linspace(x.min(), x.max(), 300)
    spl = make_interp_spline(x, y, k=3)
    y_smooth = spl(x_smooth)
    
    plt.figure(figsize=(12, 6))
    plt.plot(x_smooth, y_smooth, linewidth=2)
    plt.xlabel('Epochs')
    plt.ylabel('Cost')
    plt.title(f'Cost vs Epochs (Learning Rate = {alpha})')
    plt.yscale('log')
    
    ax = plt.gca()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:.2f}"))
    
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    plt.scatter(x[0], y[0], color='red', s=50, zorder=5, label='Start')
    plt.scatter(x[-1], y[-1], color='green', s=50, zorder=5, label='End')
    
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    X_train, y_train = load_data('Datasets/concrete/train.csv')
    X_test, y_test = load_data('Datasets/concrete/test.csv')
    
    initial_theta = np.zeros(X_train.shape[1])
    num_epochs = 10000  

    # Tuning learning rate
    learning_rates = [1.0, 0.5, 0.25, 0.125, 0.1, 0.01, 0.001, 0.0001]
    best_alpha = None
    best_cost = float('inf')
    best_theta = None
    best_cost_history = None

    for alpha in learning_rates:
        print(f"Testing learning rate: {alpha}")
        theta, cost_history = stochastic_gradient_descent(X_train, y_train, initial_theta, alpha, num_epochs)
        final_cost = cost_history[-1]
        
        if final_cost < best_cost:
            best_cost = final_cost
            best_alpha = alpha
            best_theta = theta
            best_cost_history = cost_history

    print(f"Best learning rate: {best_alpha}")
    plot_cost_history(best_cost_history, best_alpha)

    test_cost = compute_cost(X_test, y_test, best_theta)
    
    print('Optimized parameter values:', best_theta)
    print('Chosen learning rate:', best_alpha)
    print('Test set cost:', test_cost)

main()


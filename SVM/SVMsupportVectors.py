import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics.pairwise import rbf_kernel
# from google.colab import drive
# drive.mount('/content/drive')

train_data = pd.read_csv("Datasets/bank-note/train.csv", header=None)
train_data.iloc[:, -1] = train_data.iloc[:, -1].replace({1: 1, 0: -1})

def compute_kernel_matrix(X, gamma):
    return rbf_kernel(X, X, gamma=1 / gamma)

def dual_objective(alpha, y, kernel_matrix):
    return 0.5 * np.dot(alpha, np.dot(kernel_matrix * np.outer(y, y), alpha)) - np.sum(alpha)

def dual_constraint(alpha, y):
    return np.dot(alpha, y)

def train_dual_svm(X, y, C, kernel_matrix):
    n = len(y)
    bounds = [(0, C) for _ in range(n)]
    constraints = {'type': 'eq', 'fun': dual_constraint, 'args': (y,)}

    alpha_init = np.zeros(n)
    result = minimize(
        dual_objective,
        alpha_init,
        args=(y, kernel_matrix),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 500, 'ftol': 1e-9}  
    )
    return result.x

# Parameters
gamma_values = [0.01, 0.1, 0.5, 1, 5, 100]
C_values = [100/873, 500/873, 700/873] 

X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values

kernel_matrices = {gamma: compute_kernel_matrix(X_train, gamma) for gamma in gamma_values}

support_vectors = {C: {} for C in C_values}

for C in C_values:
    print(f"\nTraining with C = {C}")
    
    for gamma in gamma_values:
        kernel_matrix = kernel_matrices[gamma]
        alpha = train_dual_svm(X_train, y_train, C, kernel_matrix)
        support_vectors[C][gamma] = np.where(alpha > 1e-5)[0]  
        print(f"Gamma: {gamma}, Number of Support Vectors: {len(support_vectors[C][gamma])}")

print(f"\nOverlap between gamma values for C = {500/873}:")
    
for i in range(len(gamma_values) - 1):
    gamma_1, gamma_2 = gamma_values[i], gamma_values[i + 1]
    overlap = len(np.intersect1d(support_vectors[500/873][gamma_1], support_vectors[500/873][gamma_2], assume_unique=True))
    print(f"Overlap between gamma = {gamma_1} and gamma = {gamma_2}: {overlap}")

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import accuracy_score
# from google.colab import drive
# drive.mount('/content/drive')

train_data = pd.read_csv("Datasets/bank-note/train.csv", header=None)
test_data = pd.read_csv("Datasets/bank-note/test.csv", header=None)

train_data.iloc[:, -1] = train_data.iloc[:, -1].replace({1: 1, 0: -1})
test_data.iloc[:, -1] = test_data.iloc[:, -1].replace({1: 1, 0: -1})

X_train, y_train = train_data.iloc[:, :-1].values, train_data.iloc[:, -1].values
X_test, y_test = test_data.iloc[:, :-1].values, test_data.iloc[:, -1].values

def compute_kernel_matrix(X1, X2, gamma):
    return np.exp(-np.linalg.norm(X1[:, None] - X2[None, :], axis=2) ** 2 / gamma)

def dual_objective(alpha, kernel_matrix, y):
    return 0.5 * np.dot(alpha, np.dot(alpha, kernel_matrix * np.outer(y, y))) - np.sum(alpha)

def alpha_constraint(alpha, y):
    return np.dot(alpha, y)

def train_svm_gaussian_kernel(X, y, C, gamma):
    kernel_matrix = compute_kernel_matrix(X, X, gamma)
    num_samples = len(y)
    initial_alpha = np.zeros(num_samples)
    bounds = [(0, C) for _ in range(num_samples)]
    constraint = {'type': 'eq', 'fun': alpha_constraint, 'args': (y,)}
    result = minimize(
        dual_objective,
        initial_alpha,
        args=(kernel_matrix, y),
        method='SLSQP',
        bounds=bounds,
        constraints=constraint,
    )
    alpha = result.x

    support_vector_indices = (alpha > 1e-5) & (alpha < C)
    support_alphas = alpha[support_vector_indices]
    support_vectors = X[support_vector_indices]
    support_labels = y[support_vector_indices]

    kernel_subset = kernel_matrix[np.ix_(support_vector_indices, support_vector_indices)]
    bias = np.mean(
        support_labels - np.sum((support_alphas * support_labels)[:, None] * kernel_subset, axis=0)
    )

    return alpha, bias, support_vector_indices, kernel_matrix

def predict(X_train, X_test, alpha, y_train, bias, gamma, kernel_matrix_train_test=None):
    if kernel_matrix_train_test is None:
        kernel_matrix_train_test = compute_kernel_matrix(X_train, X_test, gamma)
    predictions = np.sign(np.dot((alpha * y_train), kernel_matrix_train_test) + bias)
    return predictions

C_values = [100 / 873, 500 / 873, 700 / 873]
gamma_values = [0.1, 0.5, 1, 5, 100]
results = []

for C in C_values:
    for gamma in gamma_values:
        alpha, bias, support_vector_indices, kernel_matrix_train = train_svm_gaussian_kernel(X_train, y_train, C, gamma)

        kernel_matrix_train_test = compute_kernel_matrix(X_train, X_test, gamma)

        train_predictions = predict(X_train, X_train, alpha, y_train, bias, gamma, kernel_matrix_train)
        test_predictions = predict(X_train, X_test, alpha, y_train, bias, gamma, kernel_matrix_train_test)

        train_error = 1 - accuracy_score(y_train, train_predictions)
        test_error = 1 - accuracy_score(y_test, test_predictions)

        results.append({
            'C': C,
            'gamma': gamma,
            'train_error': train_error,
            'test_error': test_error,
        })

results_df = pd.DataFrame(results)
print(results_df)

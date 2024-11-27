import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
# from google.colab import drive
# drive.mount('/content/drive')

train_data = pd.read_csv("Datasets/bank-note/train.csv", header=None)
test_data = pd.read_csv("Datasets/bank-note/test.csv", header=None)

train_data.iloc[:, -1] = train_data.iloc[:, -1].replace({1: 1, 0: -1})
test_data.iloc[:, -1] = test_data.iloc[:, -1].replace({1: 1, 0: -1})

X_train, y_train = train_data.iloc[:, :-1].values, train_data.iloc[:, -1].values
X_test, y_test = test_data.iloc[:, :-1].values, test_data.iloc[:, -1].values

def gaussian_kernel(x1, x2, gamma):
    distance = np.linalg.norm(x1 - x2) ** 2
    return np.exp(-distance / gamma)

def train_kernel_perceptron(X_train, y_train, gamma, max_epochs):
    num_samples = len(y_train)
    mistake_counts = np.zeros(num_samples)  

    for epoch in range(max_epochs):
        for i in range(num_samples):
            weighted_sum = sum(
                mistake_counts[j] * y_train[j] * gaussian_kernel(X_train[j], X_train[i], gamma)
                for j in range(num_samples)
            )
            if y_train[i] * weighted_sum <= 0:
                mistake_counts[i] += 1

    return mistake_counts

def predict_kernel_perceptron(X, X_train, y_train, mistake_counts, gamma):
    predictions = []
    num_samples = len(y_train)

    for x in X:
        weighted_sum = sum(
            mistake_counts[j] * y_train[j] * gaussian_kernel(X_train[j], x, gamma)
            for j in range(num_samples)
        )
        predictions.append(np.sign(weighted_sum))
    return np.array(predictions)

def evaluate_kernel_perceptron(X_train, y_train, X_test, y_test, gamma_values, max_epochs=10):
    results = []

    for gamma in gamma_values:
        mistake_counts = train_kernel_perceptron(X_train, y_train, gamma, max_epochs)

        train_predictions = predict_kernel_perceptron(X_train, X_train, y_train, mistake_counts, gamma)
        test_predictions = predict_kernel_perceptron(X_test, X_train, y_train, mistake_counts, gamma)

        train_error = 1 - accuracy_score(y_train, train_predictions)
        test_error = 1 - accuracy_score(y_test, test_predictions)

        results.append({
            'gamma': gamma,
            'train_error': train_error,
            'test_error': test_error,
        })

        print(f"Gamma: {gamma:.1f}, Train Error: {train_error:.4f}, Test Error: {test_error:.4f}")

    return results

gamma_values = [0.1, 0.5, 1, 5, 100]
evaluate_kernel_perceptron(X_train, y_train, X_test, y_test, gamma_values)

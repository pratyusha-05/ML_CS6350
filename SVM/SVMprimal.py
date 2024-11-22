import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

class SVM:
    def __init__(self, C, gamma0, a, max_epochs=100):
        self.C = C
        self.gamma0 = gamma0
        self.a = a
        self.max_epochs = max_epochs
        self.w = None
        self.objective_values = []

    def learning_rate(self, t):
        return self.gamma0 / (1 + (self.gamma0 * t / self.a))

    def _compute_objective(self, X, y):
        reg_term = 0.5 * np.sum(self.w[:-1] ** 2)  # Exclude bias term

        margins = y * (np.dot(X, self.w))
        hinge_losses = np.maximum(0, 1 - margins)
        avg_hinge_loss = np.mean(hinge_losses)

        return reg_term + self.C * avg_hinge_loss

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features + 1)

        X_bias = np.hstack([X, np.ones((n_samples, 1))])

        for epoch in range(self.max_epochs):
            indices = np.random.permutation(n_samples)
            X_shuffled = X_bias[indices]
            y_shuffled = y[indices]

            for i in range(n_samples):
                gamma_t = self.learning_rate(epoch * n_samples + i)

                if y_shuffled[i] * np.dot(X_shuffled[i], self.w) < 1:
                    grad = np.zeros_like(self.w)
                    grad[:-1] = self.w[:-1] - self.C * n_samples * y_shuffled[i] * X_shuffled[i][:-1]
                    grad[-1] = -self.C * n_samples * y_shuffled[i]  # Gradient for bias term
                else:
                    grad = np.zeros_like(self.w)
                    grad[:-1] = self.w[:-1]  # No gradient for bias term when margin is satisfied

                self.w -= gamma_t * grad

            obj_value = self._compute_objective(X_bias, y)
            self.objective_values.append(obj_value)

    def predict(self, X):
        X_bias = np.hstack([X, np.ones((X.shape[0], 1))])
        return np.sign(np.dot(X_bias, self.w))

def load_and_preprocess_data():
    train_data = pd.read_csv('Datasets/bank-note/train.csv', header=None)
    X_train = train_data.iloc[:, :-1].values
    y_train = train_data.iloc[:, -1].values

    test_data = pd.read_csv('Datasets/bank-note/test.csv', header=None)
    X_test = test_data.iloc[:, :-1].values
    y_test = test_data.iloc[:, -1].values

    y_train = 2 * y_train - 1
    y_test = 2 * y_test - 1

    return X_train, y_train, X_test, y_test

def evaluate_model(C, gamma0, a):

    X_train, y_train, X_test, y_test = load_and_preprocess_data()

    svm = SVM(C=C, gamma0=gamma0, a=a)
    svm.fit(X_train, y_train)

    train_pred = svm.predict(X_train)
    test_pred = svm.predict(X_test)

    train_error = 1 - accuracy_score(y_train, train_pred)
    test_error = 1 - accuracy_score(y_test, test_pred)

    return train_error, test_error, svm.objective_values

C_values = [100/873, 500/873, 700/873]
gamma0 = 0.1
a = 0.02

results = []
for C in C_values:
    train_error, test_error, obj_values = evaluate_model(C, gamma0, a)
    results.append({
        'C': C,
        'Train Error': train_error,
        'Test Error': test_error
    })



print("\nResults for different C values:")
print()
for result in results:
    print(f"C = {result['C']:.3f}")
    print(f"Training Error: {result['Train Error']:.4f}")
    print(f"Test Error: {result['Test Error']:.4f}")
    print()

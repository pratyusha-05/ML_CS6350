import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
# from google.colab import drive
# drive.mount('/content/drive')


class SVM:
    def __init__(self, C, gamma0, a, max_epochs=100, schedule_type='a'):
        self.C = C
        self.gamma0 = gamma0
        self.a = a
        self.max_epochs = max_epochs
        self.schedule_type = schedule_type
        self.w = None
        self.objective_values = []

    def learning_rate(self, t):
        if self.schedule_type == 'a': 
            return self.gamma0 / (1 + t)
        elif self.schedule_type == 'b': 
            return self.gamma0 / (1 + (self.gamma0 * t / self.a))
        else:
            raise ValueError("Invalid schedule type. Choose 'a' or 'b'.")

    def _compute_objective(self, X, y):
        reg_term = 0.5 * np.sum(self.w[:-1] ** 2)  

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
                    grad[-1] = -self.C * n_samples * y_shuffled[i]  
                else:
                    grad = np.zeros_like(self.w)
                    grad[:-1] = self.w[:-1]  

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

def evaluate_model(C, gamma0, a, schedule_type):
    X_train, y_train, X_test, y_test = load_and_preprocess_data()

    svm = SVM(C=C, gamma0=gamma0, a=a, schedule_type=schedule_type)
    svm.fit(X_train, y_train)

    train_pred = svm.predict(X_train)
    test_pred = svm.predict(X_test)

    train_error = 1 - accuracy_score(y_train, train_pred)
    test_error = 1 - accuracy_score(y_test, test_pred)

    return train_error, test_error, svm.w, svm.objective_values

C_values = [100/873, 500/873, 700/873]
gamma0 = 0.1
a = 0.02

results = []
for C in C_values:
    train_error_a, test_error_a, weights_a, obj_values_a = evaluate_model(C, gamma0, a, schedule_type='a')
    
    train_error_b, test_error_b, weights_b, obj_values_b = evaluate_model(C, gamma0, a, schedule_type='b')
    train_diff = abs(train_error_a - train_error_b)
    test_diff = abs(test_error_a - test_error_b)
    
    weight_diff = np.linalg.norm(weights_a - weights_b)
    bias_diff = abs(weights_a[-1] - weights_b[-1])  # Last term is bias
    
    results.append({
        'C': C,
        'Train Error Difference': train_diff,
        'Test Error': test_diff,
        'Weight Difference': weight_diff,
        'Bias Difference': bias_diff
    })

results_df = pd.DataFrame(results)
print(results_df)

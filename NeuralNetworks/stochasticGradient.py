import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, seed=42):
        np.random.seed(seed)
        self.weights = [
            np.random.randn(input_size, hidden_size),
            np.random.randn(hidden_size, output_size)
        ]
        self.biases = [
            np.random.randn(1, hidden_size),
            np.random.randn(1, output_size)
        ]

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def forward_pass(self, x):
        self.layer_inputs = []
        self.layer_outputs = [x]

        for w, b in zip(self.weights, self.biases):
            z = np.dot(self.layer_outputs[-1], w) + b
            self.layer_inputs.append(z)
            self.layer_outputs.append(self.sigmoid(z))

        return self.layer_outputs[-1]

    def backward_pass(self, x, y):
        delta = (self.layer_outputs[-1] - y) * self.sigmoid_derivative(self.layer_inputs[-1])
        gradients_w = []
        gradients_b = []

        for i in range(len(self.weights) - 1, -1, -1):
            gradient_w = np.dot(self.layer_outputs[i].T, delta)
            gradient_b = np.sum(delta, axis=0, keepdims=True)
            gradients_w.insert(0, gradient_w)
            gradients_b.insert(0, gradient_b)

            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.sigmoid_derivative(self.layer_inputs[i - 1])

        return gradients_w, gradients_b

    def update_weights(self, gradients_w, gradients_b, learning_rate):
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * gradients_w[i]
            self.biases[i] -= learning_rate * gradients_b[i]

    def compute_error(self, y_pred, y):
        predictions = (y_pred > 0.5).astype(int)
        return np.mean(predictions != y) * 100

def sgd(nn, x_train, y_train, x_test, y_test, gamma_0, d, epochs):
    for epoch in range(epochs):
        indices = np.arange(x_train.shape[0])
        np.random.shuffle(indices)
        x_train, y_train = x_train[indices], y_train[indices]

        for t in range(x_train.shape[0]):
            lr = gamma_0 / (1 + (gamma_0 / d) * t)
            nn.forward_pass(x_train[t:t+1])
            grads_w, grads_b = nn.backward_pass(x_train[t:t+1], y_train[t:t+1])
            nn.update_weights(grads_w, grads_b, lr)

    y_prediction_train = nn.forward_pass(x_train)
    y_prediction_test = nn.forward_pass(x_test)

    train_error = nn.compute_error(y_prediction_train, y_train)
    test_error = nn.compute_error(y_prediction_test, y_test)

    return train_error, test_error


train_data = pd.read_csv("Datasets/bank-note/train.csv", header=None)
test_data = pd.read_csv("Datasets/bank-note/test.csv", header=None)

x_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values.reshape(-1, 1)

x_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values.reshape(-1, 1)

gamma_0, d, epochs = 0.1, 0.001, 100
hidden_sizes = [5, 10, 25, 50, 100]

for h in hidden_sizes:
    print(f"\nTraining with hidden layer size: {h}")
    nn = NeuralNetwork(input_size=x_train.shape[1], hidden_size=h, output_size=1)

    train_error, test_error = sgd(nn, x_train, y_train, x_test, y_test, gamma_0, d, epochs)

    print(f"Training Error: {train_error:.2f}% , Test Error: {test_error:.2f}%")

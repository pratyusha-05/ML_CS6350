#back propogation

import numpy as np
import pandas as pd

class NeuralNetwork:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, seed=42):
        np.random.seed(seed)
        self.weights = [
            np.random.randn(input_size, hidden_size1), 
            np.random.randn(hidden_size1, hidden_size2),  
            np.random.randn(hidden_size2, output_size)
        ]
        self.biases = [
            np.random.randn(1, hidden_size1),
            np.random.randn(1, hidden_size2),
            np.random.randn(1, output_size)
        ]


    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))


    def sigmoid_derivative(self, z):
        sigmoid = 1 / (1 + np.exp(-z))
        return sigmoid * (1 - sigmoid)

    def forward_pass(self, x):
        self.layer_inputs = [] 
        self.layer_outputs = [x] 

        for w, b in zip(self.weights, self.biases):
            z = np.dot(self.layer_outputs[-1], w) + b
            self.layer_inputs.append(z)
            a = self.sigmoid(z)
            self.layer_outputs.append(a)

        return self.layer_inputs, self.layer_outputs

    def backward_pass(self, y):
        gradients_w = []
        gradients_b = []

        delta = (self.layer_outputs[-1] - y) * self.sigmoid_derivative(self.layer_inputs[-1])

        for i in range(len(self.weights) - 1, -1, -1):
            gradient_w = np.dot(self.layer_outputs[i].T, delta)
            gradient_b = np.sum(delta, axis=0, keepdims=True)

            gradients_w.insert(0, gradient_w)
            gradients_b.insert(0, gradient_b)

            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.sigmoid_derivative(self.layer_inputs[i - 1])

        return gradients_w, gradients_b
        
    def predict(self, x):
        i, layer_outputs = self.forward_pass(x)
        return (layer_outputs[-1] > 0.5).astype(int)


def calculate_accuracy(predictions, true_labels):
    return np.mean(predictions == true_labels) * 100


input_size = 4 
hidden_size1 = 5
hidden_size2 = 3
output_size = 1 

nn = NeuralNetwork(input_size, hidden_size1, hidden_size2, output_size)

train_data = pd.read_csv("Datasets/bank-note/train.csv", header=None)
test_data = pd.read_csv("Datasets/bank-note/test.csv", header=None)

x_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values.reshape(-1, 1)

x_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values.reshape(-1, 1)

x = x_train[0:1] 
y = y_train[0:1]

layer_inputs, layer_outputs = nn.forward_pass(x)

gradients_w, gradients_b = nn.backward_pass(y)

    
print("Gradients for weights:")
for i, grad_w in enumerate(gradients_w):
    print()
    print(f"Layer {i + 1} gradients:\n{grad_w}")
      
predictions = nn.predict(x_test)

accuracy = calculate_accuracy(predictions, y_test)
print(f"\nAccuracy on the test set: {accuracy:.2f}%")

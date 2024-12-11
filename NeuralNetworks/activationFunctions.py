import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, activation, init):
        super(NeuralNetwork, self).__init__()
        self.hidden = nn.Linear(input_size, hidden_size)
        self.output = nn.Linear(hidden_size, output_size)

        init(self.hidden.weight)
        init(self.output.weight)

        if activation == "tanh":
            self.activation = torch.tanh
        elif activation == "relu":
            self.activation = torch.relu

    def forward(self, x):
        x = self.activation(self.hidden(x))
        x = torch.sigmoid(self.output(x))
        return x


def train_model(model, optimizer, criterion, x_train, y_train, x_test, y_test, epochs):
    for epoch in range(epochs):
        y_prediction = model(x_train)
        loss = criterion(y_prediction, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        train_prediction = model(x_train)
        test_prediction = model(x_test)
        train_error = ((train_prediction > 0.5).float() != y_train).float().mean().item() * 100
        test_error = ((test_prediction > 0.5).float() != y_test).float().mean().item() * 100

    return train_error, test_error


train_data = pd.read_csv("Datasets/bank-note/train.csv", header=None)
test_data = pd.read_csv("Datasets/bank-note/test.csv", header=None)

x_train = torch.tensor(train_data.iloc[:, :-1].values, dtype=torch.float32)
y_train = torch.tensor(train_data.iloc[:, -1].values, dtype=torch.float32).unsqueeze(1)

x_test = torch.tensor(test_data.iloc[:, :-1].values, dtype=torch.float32)
y_test = torch.tensor(test_data.iloc[:, -1].values, dtype=torch.float32).unsqueeze(1)

input_size = x_train.shape[1]
output_size = 1
hidden_sizes = [5, 10, 25, 50, 100]
depths = [3, 5, 9]
activation_fns = {"tanh": nn.init.xavier_uniform_, "relu": nn.init.kaiming_uniform_}
epochs = 100
learning_rate = 1e-3

for activation, init in activation_fns.items():
    print(f"\nUsing activation: {activation}")
    for depth in depths:
        for hidden_size in hidden_sizes:
            print("Depth: ", depth, "Hidden Size: ", hidden_size, end=" -> ")

            model = NeuralNetwork(input_size, hidden_size, output_size, activation, init)
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            criterion = nn.BCELoss()

            train_error, test_error = train_model(model, optimizer, criterion, x_train, y_train, x_test, y_test, epochs)

            print("Training Error: ", round(train_error, 3), "%" , "Test Error: ", round(test_error, 3), "%")

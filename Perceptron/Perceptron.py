import numpy as np
import pandas as pd

train_data = pd.read_csv("Datasets/bank-note/train.csv", header=None)
test_data = pd.read_csv("Datasets/bank-note/test.csv", header=None)

X_train = train_data.iloc[:, :-1].to_numpy()
y_train = train_data.iloc[:, -1].to_numpy()
X_test = test_data.iloc[:, :-1].to_numpy()
y_test = test_data.iloc[:, -1].to_numpy()

learning_rate = 0.1
max_epochs = 10

weights = np.zeros(X_train.shape[1])

for epoch in range(max_epochs):
    permutation = np.random.permutation(len(X_train))
    X_train = X_train[permutation]
    y_train = y_train[permutation]
    
    l = range(len(X_train)) 
    
    for i in l:
        x_sample = X_train[i]
        y_true = y_train[i]
        
        if y_true * np.dot(weights, x_sample) <= 0:
            weights += learning_rate * y_true * x_sample

test_error_count = 0
for i, x in enumerate(X_test):
    if y_test[i] * np.dot(weights, x) <= 0:
        test_error_count += 1

avg_test_error = test_error_count / len(X_test)

print("Final Weight Vector:", weights)
print("Average Prediction Error on Test Data:", avg_test_error)

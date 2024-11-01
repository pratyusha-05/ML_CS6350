import pandas as pd
import numpy as np

train_data = pd.read_csv('Datasets/bank-note/train.csv', header=None)
test_data = pd.read_csv('Datasets/bank-note/test.csv', header=None)

X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

def average_perceptron(X_train, y_train, epochs, learning_rate):
    num_samples, num_features = X_train.shape
    weights = np.zeros(num_features) 
    average_weights = np.zeros(num_features) 
    
    for epoch in range(epochs):
        for i, x in enumerate(X_train):
            prediction = np.sign(np.dot(weights, x))
            if prediction == 0:
                prediction = -1
            
            # Update weights if the prediction is incorrect
            if prediction * y_train[i] <= 0:
                weights += learning_rate * y_train[i] * x
        
            average_weights += weights
    
    average_weights /= (epochs * num_samples)
    return average_weights

learning_rate = 0.1
epochs = 10

final_weights = average_perceptron(X_train, y_train, epochs, learning_rate)

errors = 0
for i, x in enumerate(X_test):
    prediction = np.sign(np.dot(final_weights, x))
    if prediction == 0:
        prediction = -1
    if prediction != y_test[i]:
        errors += 1

average_test_error = errors / len(X_test)

print(f"Learned Weight Vector: {final_weights}")
print(f"Average Test Error: {average_test_error:.3f}")

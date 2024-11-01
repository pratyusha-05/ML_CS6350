import pandas as pd
import numpy as np

train_data = pd.read_csv('Datasets/bank-note/train.csv', header=None)
test_data = pd.read_csv('Datasets/bank-note/test.csv', header=None)

X_train = train_data.iloc[:, :-1].values
y_train = train_data.iloc[:, -1].values
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, -1].values

learning_rate = 0.1
epochs = 10

num_samples, num_features = X_train.shape
weights = np.zeros(num_features)
unique_weight_vectors = []  
correct_counts = []  

for epoch in range(epochs):
    errors = 0
    for i, x in enumerate(X_train):
        prediction = np.sign(np.dot(weights, x))
        if prediction == 0:
            prediction = -1
        if prediction * y_train[i] <= 0:
            # Update weights if prediction is incorrect
            weights += learning_rate * y_train[i] * x
            errors += 1

    unique_weight_vectors.append(weights.copy())
    correct_counts.append(num_samples - errors)

test_error_rates = []
for weights, count in zip(unique_weight_vectors, correct_counts):
    errors = sum(
        1 for i, x in enumerate(X_test)
        if np.sign(np.dot(weights, x)) != y_test[i]
    )
    test_error_rates.append(errors / len(X_test))

avg_test_error = np.mean(test_error_rates)

for i, (weights, count) in enumerate(zip(unique_weight_vectors, correct_counts)):
    print("Weight Vector", i + 1, ":",  weights, "Correct Count:", count)

print(f"Average Test Error: {avg_test_error:.3f}")

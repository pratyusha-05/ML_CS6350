import numpy as np
from BaggedTrees import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd 
from collections import Counter

class RandomForestClassifier:
    def __init__(self, n_trees, max_features, max_depth=np.inf):
        self.n_trees = n_trees
        self.max_features = max_features
        self.trees = []

    def fit(self, X, Y):
        n_samples, n_features = X.shape
        for i in range(self.n_trees):
            selected_features = np.random.choice(n_features, self.max_features, replace=False)
            X_subset = X[:, selected_features]

            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_resampled, Y_resampled = X_subset[indices], Y[indices]

            dt = DecisionTreeClassifier(max_depth=10)
            dt.fit(X_resampled, Y_resampled)
            self.trees.append((dt, selected_features))

    def predict(self, X):
        predictions = np.zeros(X.shape[0])
        for dt, selected_features in self.trees:
            X_subset = X[:, selected_features]
            predictions += dt.predict(X_subset)
            
        return np.sign(predictions)


max_trees = 500
feature_subset_size = [2, 4, 6]

train_errors = {2: [], 4: [], 6: []}
test_errors = {2: [], 4: [], 6: []}

columns = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']

trainDf = pd.read_csv("Datasets/bank/train.csv", names=columns)
X_train = trainDf.drop('y', axis=1).values
y_train = trainDf['y'].apply(lambda x: 1 if x == 'yes' else 0).values.astype(float)

testDf = pd.read_csv("Datasets/bank/test.csv", names=columns)
X_test = testDf.drop('y', axis=1).values
y_test = testDf['y'].apply(lambda x: 1 if x == 'yes' else 0).values.astype(float)

for s in feature_subset_size:
    for num_trees in range(1, max_trees+1):
        print(num_trees)
        rf_classifier = RandomForestClassifier(num_trees, s)
        rf_classifier.fit(X_train, y_train)

        y_train_pred = rf_classifier.predict(X_train)
        y_test_pred = rf_classifier.predict(X_test)

        train_error = 1 - accuracy_score(y_train, y_train_pred)
        test_error = 1 - accuracy_score(y_test, y_test_pred)

        train_errors[s].append(train_error)
        test_errors[s].append(test_error)

plt.figure(figsize=(12, 6))
for s in feature_subset_size:
    plt.plot(range(1, max_trees+1), train_errors[s], label=f'Train Error (max_features={s})')
    plt.plot(range(1, max_trees+1), test_errors[s], label=f'Test Error (max_features={s})')

plt.xlabel('Number of Random Trees')
plt.ylabel('Error')
plt.title('Training and Test Errors vs. Number of Random Trees')
plt.legend()
plt.show()
plt.savefig('randomForest.png')


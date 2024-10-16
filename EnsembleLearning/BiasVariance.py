
from BaggedTrees import BaggedTrees, DecisionTreeClassifier
import numpy as np
from sklearn.metrics import mean_squared_error
import pandas as pd

def bias_variance_calculation(predictions, ground_truth):
    bias = np.mean(predictions) - ground_truth
    variance = np.var(predictions)
    return bias, variance

columns = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']

trainDf = pd.read_csv("Datasets/bank/train.csv", names=columns)
X_train = trainDf.drop('y', axis=1).values
y_train = trainDf['y'].apply(lambda x: 1 if x == 'yes' else 0).values.astype(float)

testDf = pd.read_csv("Datasets/bank/test.csv", names=columns)
X_test = testDf.drop('y', axis=1).values
y_test = testDf['y'].apply(lambda x: 1 if x == 'yes' else 0).values.astype(float)

#bias and variances
single_tree_biases = []
single_tree_variances = []
bagged_tree_biases = []
bagged_tree_variances = []
num_iterations = 100
num_bagged_trees = 500

for iteration in range(1, num_iterations + 1):
    n_samples = X_train.shape[0]
    
    sample_indices = np.random.choice(n_samples, size=1000, replace=False)
    X_train_resampled, y_train_resampled = X_train[sample_indices], y_train[sample_indices]

    bagged_trees_model = BaggedTrees(num_bagged_trees)
    
    bagged_trees_model.fit(X_train_resampled, y_train_resampled)

    individual_tree_predictions = np.array([tree.predict(X_test) for tree in bagged_trees_model.trees])
    
    avg_individual_tree_predictions = np.mean(individual_tree_predictions, axis=0)
    
    single_tree_bias, single_tree_variance = bias_variance_calculation(avg_individual_tree_predictions, y_test)
    single_tree_biases.append(single_tree_bias)
    single_tree_variances.append(single_tree_variance)

    bagged_tree_predictions = bagged_trees_model.predict(X_test)

    bagged_tree_bias, bagged_tree_variance = bias_variance_calculation(bagged_tree_predictions, y_test)
    bagged_tree_biases.append(bagged_tree_bias)
    bagged_tree_variances.append(bagged_tree_variance)

    
avg_bias_single_tree = np.mean(single_tree_biases)
avg_variance_single_tree = np.mean(single_tree_variances)
avg_bias_bagged_tree = np.mean(bagged_tree_biases)
avg_variance_bagged_tree = np.mean(bagged_tree_variances)

print("Average bias of single decision tree:", avg_bias_single_tree)
print("Average variance of single decision tree:", avg_variance_single_tree)
print("Average bias of bagged trees:", avg_bias_bagged_tree)
print("Average variance of bagged trees:", avg_variance_bagged_tree)


#!/bin/sh

echo "Adaboost on Bank dataset"
python3 Adaboost.py

echo "Bagged Trees on Bank dataset"
python3 BaggedTrees.py

echo "Bias and Variance calculation"
python3 BiasVariance.py

echo "Random Forest on Bank dataset"
python3 RandomForest.py

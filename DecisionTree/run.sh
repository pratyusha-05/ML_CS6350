#!/bin/sh

echo "Average Train and Test prediction errors on Car dataset"
python3 DecisionTreeCar.py

echo "Average Train and Test prediction errors on Bank dataset"
python3 DecisionTreeBank.py

echo "Average Train and Test prediction errors on Bank dataset by replacing missing values"
python3 DecisionTreeBankMissing.py

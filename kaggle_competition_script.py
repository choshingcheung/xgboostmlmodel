import os
import pandas as pd
from xgboost import XGBRegressor

# Read the data
X = pd.read_csv('train.csv', index_col='Id')
X_test_full = pd.read_csv('test.csv', index_col='Id')

# Remove rows with missing target, separate target from predictors
X.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X.SalePrice              
X.drop(['SalePrice'], axis=1, inplace=True)

# Select categorical columns with relatively low cardinality
low_cardinality_cols = [cname for cname in X.columns if X[cname].nunique() < 10 and X[cname].dtype == "object"]

# Select numeric columns
numeric_cols = [cname for cname in X.columns if X[cname].dtype in ['int64', 'float64']]

# Keep selected columns only
my_cols = low_cardinality_cols + numeric_cols
X_train = X[my_cols].copy()
X_test = X_test_full[my_cols].copy()

# One-hot encode the data
X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)
X_train, X_test = X_train.align(X_test, join='left', axis=1)

# Define the model (without early stopping)
my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, 
             early_stopping_rounds=10, n_jobs=4, max_depth=6, min_child_weight=1, subsample=0.8, colsample_bytree=0.8)

# Fit the model on the entire training data
my_model.fit(X_train, y, 
             eval_set=[(X_train, y)], 
             verbose=False)

# Get predictions on the test data
predictions = my_model.predict(X_test)

# Save predictions for submission
output = pd.DataFrame({'Id': X_test.index,
                       'SalePrice': predictions})
output.to_csv('submission.csv', index=False)

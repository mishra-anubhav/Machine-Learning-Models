#This file contains the code to perfom linear regresisson to find the price of the house based on its features

from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset and split into features X and target y
boston = load_boston()
X, y = boston.data, boston.target

# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Make predictions on the same data (for demonstration)
preds = model.predict(X)

# Evaluate the model
mse = mean_squared_error(y, preds)
r2 = r2_score(y, preds)
print(f"Training MSE: {mse:.2f}, R^2: {r2:.2f}")

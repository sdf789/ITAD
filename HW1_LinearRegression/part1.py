# Step 1: Import libraries
import numpy as np
import matplotlib.pyplot as plt

# Step 2: Generate some example data (X as inputs, y as outputs)
# Let's assume a linear relation: y = 2X + 3 + some_noise
np.random.seed(0)  # For reproducibility
X = 2 * np.random.rand(100, 1)  # 100 random points between 0 and 2
y = 4 + 3 * X + np.random.randn(100, 1)  # y = 3X + 4 + Gaussian noise

# Step 3: Prepare the data by adding a column of ones (for intercept term)
X_b = np.c_[np.ones((100, 1)), X]  # Add a column of ones to X for the bias term

# Step 4: Compute the optimal values of theta using the Normal Equation
# Theta = (X_b^T * X_b)^-1 * X_b^T * y
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

# Step 5: Print the results (slope and intercept)
print(f'Intercept: {theta_best[0][0]}')
print(f'Slope: {theta_best[1][0]}')

# Step 6: Make predictions using the computed theta
X_new = np.array([[0], [2]])  # New data points for predictions
X_new_b = np.c_[np.ones((2, 1)), X_new]  # Add bias term (column of ones)
y_predict = X_new_b.dot(theta_best)

# Step 7: Plot the results
plt.plot(X_new, y_predict, "r-", label="Predictions")  # Line of best fit
plt.plot(X, y, "b.", label="Data points")  # Original data points
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()

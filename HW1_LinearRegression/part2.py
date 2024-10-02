# Install Streamlit if you haven't already by running: pip install streamlit

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Title of the Streamlit App
st.title('Linear Regression Example with Streamlit')

# Sidebar for input parameters
st.sidebar.header('Generate Data for Linear Regression')

# Generate Random Data Controls
n_samples = st.sidebar.slider('Number of Samples', min_value=10, max_value=200, value=100)
noise_level = st.sidebar.slider('Noise Level', min_value=0.0, max_value=5.0, value=1.0)

# Generate random data
np.random.seed(42)
X = 2 * np.random.rand(n_samples, 1)
y = 4 + 3 * X + noise_level * np.random.randn(n_samples, 1)

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating the Linear Regression model
model = LinearRegression()

# Training the model
model.fit(X_train, y_train)

# Making predictions on the test set
y_pred = model.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Display results
st.write(f"Mean Squared Error: {mse:.2f}")
st.write(f"R² Score: {r2:.2f}")

# Plotting the results
plt.figure(figsize=(8,6))
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.plot(X_test, y_pred, color='red', label='Predicted', linewidth=2)
plt.title('Linear Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()

# Display the plot in the Streamlit app
st.pyplot(plt)

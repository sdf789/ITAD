# Install Streamlit and other required packages if you haven't already:
# pip install streamlit scikit-learn matplotlib

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd

# Streamlit App Title
st.title('Interactive Linear Regression')

# Sidebar controls for user inputs
st.sidebar.header('Input Parameters')

# Slider for 'a' (slope of the line)
a = st.sidebar.slider('Select the value of a (slope)', min_value=-10.0, max_value=10.0, value=1.0)

# Slider for 'c' (noise level)
c = st.sidebar.slider('Select the value of c (noise level)', min_value=0.0, max_value=100.0, value=10.0)

# Slider for the number of points (n)
n_points = st.sidebar.slider('Select the number of points (n)', min_value=10, max_value=500, value=100)

# Generate synthetic data based on user inputs
np.random.seed(42)
X = 10 * np.random.rand(n_points, 1)  # Random values of X
y = a * X + 50 + c * np.random.randn(n_points, 1)  # Linear relationship with noise

# Linear Regression model
model = LinearRegression()

# Fit the model to the data
model.fit(X, y)

# Predict values
y_pred = model.predict(X)

# Create a DataFrame to store point coordinates for display
data_points = pd.DataFrame({'X': X.flatten(), 'y (True)': y.flatten(), 'y (Predicted)': y_pred.flatten()})

# Display the table of points
st.write("Data Points:")
st.dataframe(data_points)

# Plotting the data points and the regression line
plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='blue', label='Data Points')
plt.plot(X, y_pred, color='red', label='Regression Line', linewidth=2)
plt.title(f'Linear Regression: y = {a}*x + 50 + c*random noise')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()

# Display the plot in the Streamlit app
st.pyplot(plt)

# Display the equation of the fitted line
st.write(f"Equation of the regression line: y = {model.coef_[0][0]:.2f}*x + {model.intercept_[0]:.2f}")

# Optional: Download the data points as CSV file
csv = data_points.to_csv(index=False)
st.download_button(
    label="Download data as CSV",
    data=csv,
    file_name='data_points.csv',
    mime='text/csv',
)

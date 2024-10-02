# part4.py

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Function to perform linear regression and plotting
def linear_regression(a, c, n):
    # Step 1: Generate example data
    X = np.random.rand(n) * 100  # Random X values between 0 and 100
    y = a * X + 50 + c * np.random.randn(n)  # y = a*x + 50 + noise

    # Step 2: Prepare the data
    X_b = np.c_[np.ones((n, 1)), X]  # Add a column of ones for intercept

    # Step 3: Compute the optimal values of theta using the Normal Equation
    theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

    # Step 4: Make predictions
    X_new = np.array([[0], [100]])  # New data points for predictions
    X_new_b = np.c_[np.ones((2, 1)), X_new]  # Add bias term
    y_predict = X_new_b.dot(theta_best)

    # Step 5: Plotting the results
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color='blue', label='Data points')  # Original data points
    plt.plot(X_new, y_predict, "r-", label="Regression Line")  # Regression line
    plt.xlabel("X")
    plt.ylabel("y")
    plt.title("Linear Regression")
    plt.legend()
    st.pyplot(plt)  # Use Streamlit to display the plot

    # Step 6: Show points location
    st.write("### Data Points Location:")
    data_points = np.column_stack((X, y))  # Combine X and y into one array for display
    st.dataframe(data_points)  # Display data points as a DataFrame

# Main function to run the app
def main():
    st.title("Linear Regression with Streamlit")
    st.write("This app demonstrates linear regression with randomly generated data.")

    # User inputs
    a = st.slider("Select a (slope)", -10.0, 10.0, 0.0, 0.1)
    c = st.slider("Select c (noise multiplier)", 0.0, 100.0, 0.0, 1.0)
    n = st.slider("Select number of points (n)", 10, 500, 100)

    # Automatically run regression on slider change
    linear_regression(a, c, n)

# Entry point of the application
if __name__ == "__main__":
    main()

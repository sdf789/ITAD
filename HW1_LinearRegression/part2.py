# part2.py

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# Function to perform linear regression and plotting
def linear_regression():
    # Step 1: Generate example data
    np.random.seed(0)  # For reproducibility
    X = 2 * np.random.rand(100, 1)  # 100 random points between 0 and 2
    y = 4 + 3 * X + np.random.randn(100, 1)  # y = 3X + 4 + Gaussian noise

    # Step 2: Prepare the data
    X_b = np.c_[np.ones((100, 1)), X]  # Add a column of ones for intercept

    # Step 3: Compute the optimal values of theta using the Normal Equation
    theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

    # Step 4: Print the results
    st.write(f'**Intercept:** {theta_best[0][0]:.2f}')
    st.write(f'**Slope:** {theta_best[1][0]:.2f}')

    # Step 5: Make predictions
    X_new = np.array([[0], [2]])  # New data points for predictions
    X_new_b = np.c_[np.ones((2, 1)), X_new]  # Add bias term
    y_predict = X_new_b.dot(theta_best)

    # Step 6: Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(X_new, y_predict, "r-", label="Predictions")  # Line of best fit
    plt.plot(X, y, "b.", label="Data points")  # Original data points
    plt.xlabel("X")
    plt.ylabel("y")
    plt.title("Linear Regression")
    plt.legend()
    st.pyplot(plt)  # Use Streamlit to display the plot

# Main function to run the app
def main():
    st.title("Linear Regression with Streamlit")
    st.write("This app demonstrates linear regression using generated data.")
    linear_regression()

# Entry point of the application
if __name__ == "__main__":
    main()

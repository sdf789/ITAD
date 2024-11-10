import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Generate a simple 1D dataset
np.random.seed(0)
X = np.random.randn(100, 1)  # 100 points in 1D space
y = (X > 0).astype(int).ravel()  # Classify points based on their sign (0 for negative, 1 for positive)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Initialize logistic regression and SVM models with different kernels
log_reg = LogisticRegression()
svm_linear = SVC(kernel='linear')
svm_poly = SVC(kernel='poly', degree=3)  # Polynomial kernel with degree 3
svm_rbf = SVC(kernel='rbf', gamma='scale')  # RBF kernel with default gamma

# Train the models
log_reg.fit(X_train, y_train)
svm_linear.fit(X_train, y_train)
svm_poly.fit(X_train, y_train)
svm_rbf.fit(X_train, y_train)

# Predict on the test set
y_pred_log_reg = log_reg.predict(X_test)
y_pred_svm_linear = svm_linear.predict(X_test)
y_pred_svm_poly = svm_poly.predict(X_test)
y_pred_svm_rbf = svm_rbf.predict(X_test)

# Calculate accuracy
acc_log_reg = accuracy_score(y_test, y_pred_log_reg)
acc_svm_linear = accuracy_score(y_test, y_pred_svm_linear)
acc_svm_poly = accuracy_score(y_test, y_pred_svm_poly)
acc_svm_rbf = accuracy_score(y_test, y_pred_svm_rbf)

# Print the accuracies
print(f"Logistic Regression Accuracy: {acc_log_reg:.2f}")
print(f"SVM Linear Kernel Accuracy: {acc_svm_linear:.2f}")
print(f"SVM Polynomial Kernel Accuracy: {acc_svm_poly:.2f}")
print(f"SVM RBF Kernel Accuracy: {acc_svm_rbf:.2f}")

# Plotting the decision boundaries
x_values = np.linspace(-3, 3, 500).reshape(-1, 1)
y_values_log_reg = log_reg.predict_proba(x_values)[:, 1]
y_values_svm_linear = svm_linear.decision_function(x_values)
y_values_svm_poly = svm_poly.decision_function(x_values)
y_values_svm_rbf = svm_rbf.decision_function(x_values)

plt.figure(figsize=(10, 6))
plt.plot(X[y == 0], y[y == 0], "bo", label="Class 0")
plt.plot(X[y == 1], y[y == 1], "ro", label="Class 1")

# Plot each model's decision boundary
plt.plot(x_values, y_values_log_reg, "b-", label="Logistic Regression")
plt.plot(x_values, y_values_svm_linear, "g--", label="SVM Linear Kernel")
plt.plot(x_values, y_values_svm_poly, "m-.", label="SVM Polynomial Kernel")
plt.plot(x_values, y_values_svm_rbf, "r:", label="SVM RBF Kernel")

# Set the Y-axis limits to increase the distance between boundaries
plt.ylim(-2, 2)

plt.axhline(0.5, color="grey", linestyle=":")  # Threshold for logistic regression
plt.axhline(0, color="grey", linestyle="--")   # Threshold for SVM
plt.legend()
plt.xlabel("X")
plt.ylabel("Decision Boundary / Probability")
plt.title("Logistic Regression vs SVM with Different Kernels on 1D Data")
plt.show()

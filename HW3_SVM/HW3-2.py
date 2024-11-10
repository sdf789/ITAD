import numpy as np
import streamlit as st
from sklearn.svm import LinearSVC
import plotly.graph_objects as go

# Streamlit App Title
st.title('3D Scatter Plot with Adjustable Distance Threshold')

# Sidebar slider for adjusting distance threshold
distance_threshold = st.sidebar.slider('Distance Threshold', min_value=0.1, max_value=10.0, value=4.0, step=0.1)

# Generate data
np.random.seed(0)
num_points = 600
mean = 0
variance = 10

# Randomly generate points
x1 = np.random.normal(mean, np.sqrt(variance), num_points)
x2 = np.random.normal(mean, np.sqrt(variance), num_points)

# Calculate distance and assign labels based on adjustable threshold
distances = np.sqrt(x1**2 + x2**2)
Y = np.where(distances < distance_threshold, 0, 1)

# Define Gaussian function for z-axis values
def gaussian_function(x1, x2):
    return np.exp(-0.1 * (x1**2 + x2**2))

# Calculate z-axis (height)
x3 = gaussian_function(x1, x2)
X = np.column_stack((x1, x2, x3))

# Train Linear SVM classifier
clf = LinearSVC(random_state=0, max_iter=10000)
clf.fit(X, Y)
coef = clf.coef_[0]
intercept = clf.intercept_

# Create scatter plot with Plotly
fig = go.Figure()

# Plot points for each class
fig.add_trace(go.Scatter3d(
    x=x1[Y == 0], y=x2[Y == 0], z=x3[Y == 0],
    mode='markers',
    marker=dict(color='blue', size=5),
    name='Y=0 (distance < threshold)'
))
fig.add_trace(go.Scatter3d(
    x=x1[Y == 1], y=x2[Y == 1], z=x3[Y == 1],
    mode='markers',
    marker=dict(color='red', size=5),
    name='Y=1 (distance >= threshold)'
))

# Create meshgrid for hyperplane
xx, yy = np.meshgrid(np.linspace(min(x1), max(x1), 10), np.linspace(min(x2), max(x2), 10))
zz = (-coef[0] * xx - coef[1] * yy - intercept) / coef[2]

# Add the hyperplane as a surface
fig.add_trace(go.Surface(
    x=xx, y=yy, z=zz,
    colorscale='gray',
    opacity=0.5,
    showscale=False,
    name='Separating Hyperplane'
))

# Update layout for larger dimensions
fig.update_layout(
    scene=dict(
        xaxis_title='x1',
        yaxis_title='x2',
        zaxis_title='x3'
    ),
    title='3D Scatter Plot with Adjustable Distance Threshold and Rotatable View',
    width=1200,  # Set width of the figure
    height=800   # Set height of the figure
)

# Display the plot using Streamlit with container width
st.plotly_chart(fig, use_container_width=True)

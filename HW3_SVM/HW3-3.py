import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import plotly.graph_objects as go

# 定義三角形範圍內的數據生成函數
def generate_triangle_data(n_samples=300):
    np.random.seed(0)
    vertices = np.array([[0, 0], [4, 0], [2, 3]])
    X = []
    while len(X) < n_samples:
        x = np.random.uniform(0, 4)
        y = np.random.uniform(0, 3)
        a = ((vertices[1, 1] - vertices[2, 1]) * (x - vertices[2, 0]) +
             (vertices[2, 0] - vertices[1, 0]) * (y - vertices[2, 1])) / \
            ((vertices[1, 1] - vertices[2, 1]) * (vertices[0, 0] - vertices[2, 0]) +
             (vertices[2, 0] - vertices[1, 0]) * (vertices[0, 1] - vertices[2, 1]))
        b = ((vertices[2, 1] - vertices[0, 1]) * (x - vertices[2, 0]) +
             (vertices[0, 0] - vertices[2, 0]) * (y - vertices[2, 1])) / \
            ((vertices[1, 1] - vertices[2, 1]) * (vertices[0, 0] - vertices[2, 0]) +
             (vertices[2, 0] - vertices[1, 0]) * (vertices[0, 1] - vertices[2, 1]))
        c = 1 - a - b
        if a >= 0 and b >= 0 and c >= 0:
            X.append([x, y])
    X = np.array(X)
    labels = np.where(X[:, 1] > 1.5, 1, 0)
    return X, labels

# 高斯函數，用於生成第三維度的高度
def gaussian_function(x, y, sigma=1.0):
    return np.exp(-(x**2 + y**2) / (2 * sigma**2))

# 訓練 SVM 模型
def train_svm(X, y, kernel):
    model = SVC(kernel=kernel, gamma='auto')
    model.fit(X, y)
    return model

# 使用 Plotly 畫出 3D 的 SVM 決策邊界
def plot_3d_decision_boundary(X, y, model, sigma=1.0):
    z = gaussian_function(X[:, 0], X[:, 1], sigma=sigma)
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    zz = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    zz = zz.reshape(xx.shape)

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=X[y == 0, 0], y=X[y == 0, 1], z=z[y == 0],
        mode='markers',
        marker=dict(color='blue', size=5),
        name='Class 0'
    ))
    fig.add_trace(go.Scatter3d(
        x=X[y == 1, 0], y=X[y == 1, 1], z=z[y == 1],
        mode='markers',
        marker=dict(color='red', size=5),
        name='Class 1'
    ))
    fig.add_trace(go.Surface(
        x=xx, y=yy, z=zz,
        colorscale='gray',
        opacity=0.3,
        showscale=False,
        name='Decision Boundary'
    ))
    fig.update_layout(
        scene=dict(
            xaxis_title='Feature 1',
            yaxis_title='Feature 2',
            zaxis_title='Height (Gaussian Value)',
            zaxis=dict(range=[z.min(), z.max() + 1])
        ),
        title="3D Decision Boundary with Gaussian Heights",
        width=1400,
        height=1000,
    )
    st.plotly_chart(fig)

# Streamlit 介面
st.title("Triangle Data Distribution with 3D SVM Decision Boundary")

# 選擇 SVM kernel
kernel = st.selectbox("Select SVM Kernel", ("linear", "poly", "rbf", "sigmoid"))

# 生成並顯示三角形分布數據
X, y = generate_triangle_data()
st.write("## Generated Triangle Data Distribution")
fig, ax = plt.subplots()
ax.scatter(X[y == 0, 0], X[y == 0, 1], color='blue', label='Class 0', alpha=0.5)
ax.scatter(X[y == 1, 0], X[y == 1, 1], color='red', label='Class 1', alpha=0.5)
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.set_title("Triangle Data Distribution")
st.pyplot(fig)

# 訓練 SVM 模型
model = train_svm(X, y, kernel)
st.write(f"## SVM Model Decision Boundary (3D) - Kernel: {kernel}")

# 調整高斯函數的 sigma 值
sigma = st.slider('Gaussian Sigma', min_value=0.1, max_value=5.0, value=1.0, step=0.1)

# 繪製 3D 決策邊界圖
plot_3d_decision_boundary(X, y, model, sigma)

import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from sklearn.svm import SVC
import plotly.graph_objects as go
from sklearn.datasets import make_circles

# 定義高斯函數，根據 x 和 y 生成第三維度 z 的數值
def gaussian_function(x, y, sigma=1.0):
    return np.exp(-(x**2 + y**2) / (2 * sigma**2))

# 生成圓形數據
def generate_circular_data(n_samples=300, noise=0.05):
    X, y = make_circles(n_samples=n_samples, factor=0.5, noise=noise)
    return X, y

# 訓練 SVM 模型
def train_svm(X, y):
    model = SVC(kernel='rbf', gamma='auto')
    model.fit(X, y)
    return model

# 使用 Plotly 畫出 3D 的 SVM 決策邊界
def plot_3d_decision_boundary(X, y, model, sigma=1.0):
    # 使用高斯函數計算每個數據點的 z 值
    z = gaussian_function(X[:, 0], X[:, 1], sigma=sigma)

    # 生成網格點以繪製圖形
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    
    # 根據網格點計算 SVM 決策函數
    zz = model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    zz = zz.reshape(xx.shape)

    # 創建 Plotly 圖形
    fig = go.Figure()

    # 繪製數據點，並使用高斯函數計算的 z 值作為高度
    fig.add_trace(go.Scatter3d(
        x=X[y == 0, 0], y=X[y == 0, 1], z=z[y == 0],  # 使用高斯函數生成的高度作為 z 值
        mode='markers',
        marker=dict(color='blue', size=5),
        name='Class 0'
    ))
    fig.add_trace(go.Scatter3d(
        x=X[y == 1, 0], y=X[y == 1, 1], z=z[y == 1],  # 使用高斯函數生成的高度作為 z 值
        mode='markers',
        marker=dict(color='red', size=5),
        name='Class 1'
    ))

    # 繪製決策邊界
    fig.add_trace(go.Surface(
        x=xx, y=yy, z=zz,
        colorscale='gray',
        opacity=0.3,
        showscale=False,
        name='Decision Boundary'
    ))

    # 更新圖表布局
    fig.update_layout(
        scene=dict(
            xaxis_title='Feature 1',
            yaxis_title='Feature 2',
            zaxis_title='Height (Gaussian Value)',
        ),
        title="3D Decision Boundary with Gaussian Heights",
        width=1400,  # 設置圖表的寬度
        height=1000,  # 設置圖表的高度
    )
    # 設定 Z 軸範圍大於 0
    fig.update_layout(scene=dict(zaxis=dict(range=[0, 2.5])))
    # 顯示 Plotly 圖表
    st.plotly_chart(fig)

# Streamlit 介面
st.title("2D SVM with 3D Decision Boundary and Gaussian Heights")

# 生成並顯示數據
X, y = generate_circular_data()
st.write("## Generated Circular Data")
fig, ax = plt.subplots()
ax.scatter(X[y == 0, 0], X[y == 0, 1], color='blue', label='Class 0', alpha=0.5)
ax.scatter(X[y == 1, 0], X[y == 1, 1], color='red', label='Class 1', alpha=0.5)
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.set_title("Circular Data Distribution")
st.pyplot(fig)

# 訓練 SVM 模型
model = train_svm(X, y)
st.write("## SVM Model Decision Boundary (3D)")

# 可以通過滑塊來調整高斯函數的 sigma 值
sigma = st.slider('Gaussian Sigma', min_value=0.1, max_value=3.0, value=0.5, step=0.1)

# 繪製 3D 圖形
plot_3d_decision_boundary(X, y, model, sigma)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import kneighbors_graph
from scipy.linalg import eigh


def NPE(X, n_neighbors=5, n_components=2):
    """
    执行邻域保持嵌入算法。
    :param X: 输入数据，形状为 (样本数量, 特征维数)
    :param n_neighbors: 每个点的邻居数量
    :param n_components: 降维后的维数
    :return: 降维后的数据
    """
    # 构建邻接图
    connectivity = kneighbors_graph(X, n_neighbors=n_neighbors, mode='connectivity', include_self=True)
    adjacency_matrix = connectivity.toarray()

    # 计算权重矩阵W
    n_samples = X.shape[0]
    W = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        neighbors = np.where(adjacency_matrix[i, :] > 0)[0]
        if len(neighbors) > 1:
            Z = X[neighbors, :] - X[i]
            C = Z @ Z.T
            C_inv = np.linalg.pinv(C)
            w = C_inv.sum(axis=1) / C_inv.sum()
            W[i, neighbors] = w

    # 计算投影矩阵
    M = np.eye(n_samples) - W - W.T + np.dot(W.T, W)
    eigenvalues, eigenvectors = eigh(M)

    # 选择具有最小特征值的前n_components个特征向量
    index = np.argsort(eigenvalues)[1:n_components + 1]
    return eigenvectors[:, index]


# 创建三维的样本数据
np.random.seed(0)
X = np.random.rand(200, 3)  # 200个点，每个点3维

# 应用NPE，将数据降至两维
n_components = 2
transformed_data = NPE(X, n_neighbors=10, n_components=n_components)

# 绘制原始数据和变换后的数据
fig = plt.figure(figsize=(12, 6))

# 原始数据的3D散点图
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c='blue', label='Original Data')
ax1.set_title("Original Data")
ax1.set_xlabel("X1")
ax1.set_ylabel("X2")
ax1.set_zlabel("X3")

# 变换后数据的2D散点图
ax2 = fig.add_subplot(122)
ax2.scatter(transformed_data[:, 0], transformed_data[:, 1], c='red', label='Transformed Data')
ax2.set_title("Data After NPE")
ax2.set_xlabel("Component 1")
ax2.set_ylabel("Component 2")
ax2.legend()

plt.show()

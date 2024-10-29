import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import laplacian


def locality_preserving_projection(X, n_neighbors=5, n_components=1):
    # 计算邻接矩阵
    adjacency_matrix = kneighbors_graph(X, n_neighbors=n_neighbors, include_self=True).toarray()

    # 计算拉普拉斯矩阵
    L = laplacian(adjacency_matrix, normed=True)

    # 计算X' * L * X
    XLXT = X.T @ L @ X

    # 特征值分解
    eigenvalues, eigenvectors = np.linalg.eigh(XLXT)

    # 取最小的n_components个特征值对应的特征向量
    P = eigenvectors[:, :n_components]

    # 降维结果
    Y = X @ P

    return Y, P


# 生成数据
X, y = make_moons(n_samples=100, noise=0.07, random_state=0)

# 应用LPP并获取投影矩阵P
Y, P = locality_preserving_projection(X, n_neighbors=10, n_components=1)

# 可视化结果
plt.figure(figsize=(18, 5))
plt.subplot(131)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
plt.title("Original Data")

plt.subplot(132)
plt.scatter(Y, np.zeros_like(Y), c=y, cmap=plt.cm.Spectral)
plt.title("LPP Reduced Data")

plt.subplot(133)
plt.imshow(P, aspect='auto', cmap=plt.cm.Spectral)
plt.colorbar()
plt.title("Projection Matrix P")
plt.show()

print("Projection Matrix P:\n", P)

import numpy as np
import matplotlib.pyplot as plt


def locality_preserving_projection(X, W):
    # 计算度矩阵D
    D = np.diag(W.sum(axis=1))

    # 计算拉普拉斯矩阵L
    L = D - W

    # 计算X' * L * X
    XLXT = X.T @ L @ X

    # 特征值分解
    eigenvalues, eigenvectors = np.linalg.eigh(XLXT)

    # 取最小非零特征值对应的特征向量（通常是第二小的，因为最小的特征值对应的特征向量通常是平凡解）
    index = np.argsort(eigenvalues)[1]  # 取第二小的特征值对应的索引
    P = eigenvectors[:, index]

    # 降维结果
    Y = X @ P

    return Y, P


# 定义数据点
X = np.array([[1, 2], [2, 3], [10, 11]])

# 定义权重矩阵
W = np.array([[0, 1, 0],  # x1 和 x2 是邻居
              [1, 0, 0],  # x2 和 x1 是邻居
              [0, 0, 0]])  # x3 没有邻居

# 应用LPP并获取投影矩阵P
Y, P = locality_preserving_projection(X, W)

# 可视化结果
print("Projection Vector P:\n", P)
plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.scatter(X[:, 0], X[:, 1], color='blue')
plt.title("Original Data")
plt.xlabel("x1")
plt.ylabel("x2")
for i, txt in enumerate(['x1', 'x2', 'x3']):
    plt.annotate(txt, (X[i, 0], X[i, 1]))

plt.subplot(122)
plt.scatter(Y, np.zeros_like(Y), color='red')
plt.title("LPP Reduced Data")
plt.xlabel("Projected Dimension")
for i, txt in enumerate(['x1', 'x2', 'x3']):
    plt.annotate(txt, (Y[i], 0))
plt.tight_layout()
plt.show()



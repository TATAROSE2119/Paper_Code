import cupy as cp
import numpy as np
import scipy.io  # 保留 scipy.io 用于加载数据
from sklearn.metrics import pairwise_distances
from scipy.sparse import csr_matrix

# 载入.mat文件
data = scipy.io.loadmat('TE_data.mat')  # 替换为实际的文件路径
print("成功加载 .mat 文件。")

# 选择训练数据和测试数据
X_train = data['d00'].T  # 训练数据，通常用于训练模型
X_test = data['d01_te'].T  # 测试数据，用于验证模型效果
print("成功选择训练和测试数据。")

# 标准化训练数据
def standardize_data(X):
    """
    标准化数据
    X: 输入数据
    返回标准化数据以及平均值和标准差
    """
    mean_X = cp.mean(X, axis=0)
    std_X = cp.std(X, axis=0, ddof=1)
    X_std = (X - mean_X) / std_X
    print("数据标准化完成。")
    print(f"训练数据的维度为：{X_train.shape}")
    return X_std, mean_X, std_X

X_train_std, mean_X_train, std_X_train = standardize_data(X_train)

# 实现奇异值阈值化操作
def singular_value_thresholding(A, tau):
    """
    执行奇异值阈值化操作
    A: 输入矩阵
    tau: 阈值，这里是 1/mu
    """
    U, s, V = cp.linalg.svd(A, full_matrices=False)
    s_thresholded = cp.maximum(s - tau, 0)
    print(f"执行奇异值阈值化，tau={tau}。")
    return U @ np.diag(s_thresholded) @ V

# 更新 Q
def update_Q(Z, Y2, mu):
    """
    根据提供的公式更新 Q
    """
    print("正在更新 Q。")
    return singular_value_thresholding(Z + Y2 / mu, 1 / mu)

# 更新 Z
def update_Z(X, E, Q, P, Y1, Y2, mu, beta):
    """
    根据给定的公式更新 Z
    """
    m, n = X.shape
    print("正在更新 Z。")
    # 构造公式中的矩阵
    A = beta * X.T @ P @ P.T @ X + mu* X.T@X+mu * cp.eye(n)  # 添加单位矩阵乘以mu以增加数值稳定性
    B = beta * X.T @ P @ P.T @ X + mu * X.T @ X - mu *X.T @ E + mu * Q + X.T @ Y1 - Y2

    # 解线性系统求解 Z
    Z = cp.linalg.solve(A, B)
    print("Z 更新完成。")
    return Z

# 更新 E
def update_E(X, Z, Y1, mu, alpha):
    """
    根据提供的公式更新 E
    """
    m, n = X.shape
    V = X - X @ Z + Y1 / mu
    E = cp.zeros_like(V)
    print("正在更新 E。")

    for i in range(n):
        norm_Vi = cp.linalg.norm(V[:, i])
        if norm_Vi > alpha:
            E[:, i] = (norm_Vi - alpha) / norm_Vi * V[:, i]

    print("E 更新完成。")
    return E

# 计算权重矩阵 W
def compute_weight_matrix(X, t, k=7):
    # 计算点间欧式距离
    dists = pairwise_distances(X.T)
    # 创建k-近邻图
    cp.fill_diagonal(dists, cp.inf)
    neighbors = cp.argsort(dists, axis=1)[:, :k]

    # 初始化权重矩阵
    W = cp.zeros_like(dists)
    for i in range(X.T.shape[0]):
        for j in neighbors[i]:
            W[i, j] = np.exp(-dists[i, j] ** 2 / t)
            W[j, i] = W[i, j]  # 确保W是对称的

    print("权重矩阵 W 计算完成。")
    return csr_matrix(W)

# 计算拉普拉斯矩阵 L
def compute_laplacian_matrix(X, t, k=7):
    # 计算权重矩阵 W
    W = compute_weight_matrix(X, t, k)
    # 计算度矩阵 D
    D = cp.diag(cp.array(W.sum(axis=1)).flatten())
    # 计算拉普拉斯矩阵 L
    L = D - W
    print("拉普拉斯矩阵 L 计算完成。")
    return L

# 更新 P
def update_P(X, Z, L, beta, gamma):
    """
    使用特征值分解来更新 P
    """
    n = X.shape[1]
    I = cp.eye(n)
    G = (Z - I) @ (Z - I).T
    A = X @ (beta * G + gamma * L) @ X.T

    # 计算特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eigh(A)

    # 选取最大的特征值对应的特征向量
    idx = cp.argsort(eigenvalues)[::-1]  # 降序排列特征值
    P = eigenvectors[:, idx[:n]]  # 选取对应最大特征值的特征向量
    print("P 使用特征值分解更新完成。")
    return P

# 更新拉格朗斯乘子 Y 和罟法参数 mu
def update_Y_and_mu(Y1, Y2, mu, X, Z, E, Q, rho, mu_max):
    """
    更新拉格朗斯乘子和罟法参数
    """
    # 更新 Y1 和 Y2
    Y1_new = Y1 + mu * (X - X @ Z - E)
    Y2_new = Y2 + mu * (Z - Q)

    # 更新 mu
    mu_new = min(rho * mu, mu_max)
    print(f"更新拉格朗斯乘子和 mu （新 mu: {mu_new}）。")
    return Y1_new, Y2_new, mu_new

# LRJE 方法
def lrje(X, alpha, beta, gamma, rho=1.1, mu_max=1e6, max_iter=50, eps=1e-3):
    m, n = X.shape
    Z = Q = cp.zeros((n, n))
    E= cp.zeros((m, n))
    Y1 = np.zeros((m, n))
    Y2 = np.zeros((n, n))
    P = np.zeros((m, m))
    mu = 1e-6

    for i in range(max_iter):
        print(f"第 {i+1} 次迭代开始。")
        Q_old = Q.copy()
        Z_old = Z.copy()
        P_old = P.copy()

        # 计算拉普拉斯矩阵
        L = compute_laplacian_matrix(X, t=1.0, k=7)

        # 进行各个矩阵的更新
        Q = update_Q(Z, Y2, mu)
        Z = update_Z(X, E, Q, P, Y1, Y2, mu, beta)
        E = update_E(X, Z, Y1, mu, alpha)
        P = update_P(X, Z, L, beta, gamma)
        Y1, Y2, mu = update_Y_and_mu(Y1, Y2, mu, X, Z, E, Q, rho, mu_max)

        # 检查收效条件
        if np.linalg.norm(X - X @ Z - E, np.inf) < eps and np.linalg.norm(Z - Q, np.inf) < eps and np.linalg.norm(P.T @ P - np.eye(n), np.inf) < eps:
            print(f"在第 {i+1} 次迭代时收敛。")
            break
        else:
            print(f"第 {i+1} 次迭代未收敛。")

    return P  # 返回投影矩阵 P

# 使用 LRJE 方法计算 P
print("开始使用 LRJE 方法计算 P。")
P = lrje(X_train_std, alpha=0.5, beta=0.5, gamma=0.5)  # 计算投影矩阵 P
print("P 计算完成。")
Y_train = P.T @ X_train_std  # 投影到特征空间
print("训练数据投影到特征空间完成。")

# 计算协方并矩阵的反，用于后续计算 T^2 统计量
Lambda = cp.cov(Y_train, rowvar=False)
Lambda_inv = cp.linalg.inv(Lambda)
print("协方差矩阵及其逆计算完成。")

# 处理新样本
def process_new_sample(X_new, P, Lambda_inv):
    """
    处理新的样本，返回 T^2 和 SPE 统计量
    """
    # 标准化新数据
    mean_X_new = np.mean(X_new, axis=0)
    std_X_new = np.std(X_new, axis=0, ddof=1)
    X_new_std = (X_new - mean_X_new) / std_X_new
    print(f"新数据标准化完成，使用新数据的平均值和标准差，标准化后的维度为：{X_new_std.shape}")
    # 投影到特征空间
    y_new = P.T @ X_new_std
    print(f"新数据投影到特征空间完成，投影后的维度为：{y_new.shape}")
    # 重构特征部分
    X_hat_new = P @ y_new
    print(f"新数据的特征部分重构完成，重构后的维度为：{X_hat_new.shape}")
    # 残差部分
    X_tilde_new = X_new_std - X_hat_new
    print(f"新数据的残差部分计算完成，残差的维度为：{X_tilde_new.shape}")

    # 计算 T^2 和 SPE 统计量
    T2 = y_new.T @ Lambda_inv @ y_new
    SPE = cp.sum(X_tilde_new**2)
    print(f"处理新样本完成：T^2={T2}, SPE={SPE}")
    return T2, SPE

# 假设 X_new 是新的观测数据
print("处理新的测试样本。")
T2_new, SPE_new = process_new_sample(X_test, P, mean_X_train, std_X_train, Lambda_inv)
print(f"新样本的 T^2 统计量: {T2_new}")
print(f"新样本的 SPE 统计量: {SPE_new}")

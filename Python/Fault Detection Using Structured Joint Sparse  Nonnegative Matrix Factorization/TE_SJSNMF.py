import numpy as np
from scipy.linalg import eigh, inv
import matplotlib
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import kneighbors_graph
import matplotlib.pyplot as plt

# 计算邻接矩阵（使用高斯核或K近邻）
def compute_adjacency_matrix(X, method='rbf', gamma=0.5, n_neighbors=5):
    if method == 'rbf':
        # 使用高斯核构造相似度矩阵（邻接矩阵）
        A = rbf_kernel(X, gamma=gamma)
    elif method == 'knn':
        # 使用K近邻构造相似度矩阵（邻接矩阵）
        A = kneighbors_graph(X, n_neighbors=n_neighbors, mode='connectivity').toarray()
    else:
        raise ValueError("Method should be 'rbf' or 'knn'.")
    return A

# 计算拉普拉斯矩阵
def compute_laplacian_matrix(X, method='rbf', gamma=0.5, n_neighbors=5, laplacian_type='unnormalized'):
    # 计算邻接矩阵 A
    X = X.T
    A = compute_adjacency_matrix(X, method=method, gamma=gamma, n_neighbors=n_neighbors)

    # 计算度矩阵 D
    D = np.diag(np.sum(A, axis=1))  # axis=1表示按行求和

    # 计算拉普拉斯矩阵
    if laplacian_type == 'unnormalized':
        L = D - A
    elif laplacian_type == 'normalized_sym':
        # 对称归一化拉普拉斯矩阵
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D) + 1e-10))  # 防止除以0
        L = np.eye(D.shape[0]) - D_inv_sqrt @ A @ D_inv_sqrt
    elif laplacian_type == 'normalized_rw':
        # 非对称归一化拉普拉斯矩阵
        D_inv = np.diag(1.0 / (np.diag(D) + 1e-10))  # 防止除以0
        L = np.eye(D.shape[0]) - D_inv @ A
    else:
        raise ValueError("Laplacian type should be 'unnormalized', 'normalized_sym', or 'normalized_rw'.")

    return L


def initialize_parameters(X, rank):
    # 初始化矩阵 W, H, U, Y, A, B 为非负随机值
    W = np.random.rand(X.shape[0], rank)
    H = np.random.rand(rank, X.shape[1])
    U = np.random.rand(rank, X.shape[1])
    Y = np.random.rand(X.shape[0], X.shape[1])
    A = np.random.rand(X.shape[0], X.shape[1])
    B = np.random.rand(rank, X.shape[1])
    return W, H, U, Y, A, B

# 行稀疏投影操作
def row_sparse_projection(U_half, s):
    row_norms = np.linalg.norm(U_half, axis=1)  # 计算每行的 l2 范数
    if np.count_nonzero(row_norms) <= s:
        return U_half  # 如果非零行数小于等于 s，则直接返回
    else:
        # 找到具有最大 l2 范数的前 s 行
        top_s_indices = np.argsort(row_norms)[-s:]
        U_projected = np.zeros_like(U_half)
        U_projected[top_s_indices, :] = U_half[top_s_indices, :]
        return U_projected

def SJSNMF(X, L, rank, beta1, beta2, lam, s, max_iter, tol=1e-4):
    W, H, U, Y, A, B = initialize_parameters(X, rank)
    k = 0
    while k < max_iter:
        # 保存旧值以检查收敛性
        W_old, H_old = W.copy(), H.copy()
        # Step 1: 更新 W
        H_k_H_k_T = H @ H.T
        Y_k_A_k_term = Y + A / beta1
        W_half = (Y_k_A_k_term @ H.T) @ np.linalg.pinv(H_k_H_k_T)  # 计算 W_{k+1/2}
        W = np.maximum(W_half, 0)  # 投影到非负区域
        # Step 2: 更新 H
        left_term = (2 * lam * L + beta1 * W.T @ W + beta2 * np.eye(rank))
        right_term = (beta1 * W.T @ Y + beta2 * U + W.T @ A + B)
        H = np.linalg.solve(left_term, right_term)  # 通过求解线性方程组更新 H
        # Step 3: 更新 U
        U_half = np.maximum(H - B / beta2, 0)  # 计算 U_{k+1/2}，确保非负
        U = row_sparse_projection(U_half, s)  # 进行稀疏行投影
        # Step 4: 更新 Y
        Y = (1 / (1 + beta1)) * (X + beta1 * W @ H - A)  # 根据公式(22)更新 Y
        # Step 5: 更新 A
        A = A - beta1 * (W @ H - Y)
        # Step 6: 更新 B
        B = B - beta2 * (H - U)
        # 检查收敛性
        if np.linalg.norm(W - W_old) < tol and np.linalg.norm(H - H_old) < tol:
            break
        k += 1
    return W, H

# 计算重构矩阵 H 和 X
# 根据公式(23)和(24)
def reconstruct_H_and_X(W, X):
    # 重构系数矩阵 H_hat
    H_hat = inv(W.T @ W) @ W.T @ X
    # 重构矩阵 X_hat
    X_hat = W @ H_hat
    return H_hat, X_hat

# 计算 T2 和 SPE 统计量
def calculate_statistics(X, X_hat, H_hat):
    # 计算 T2 统计量
    T2 = np.diag(H_hat.T @ H_hat)
    # 计算 SPE 统计量
    SPE = np.sum((X - X_hat) ** 2, axis=0)
    return T2, SPE

# 绘制 T2 和 SPE 统计量的变化趋势
def plot_statistics(T2, SPE):
    plt.figure()
    plt.plot(T2, label='T^2 Statistic')
    plt.xlabel('Sample Index')
    plt.ylabel('T^2 Value')
    plt.legend()
    plt.title('T^2 Statistic Over Time')
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(SPE, label='SPE Statistic', color='red')
    plt.xlabel('Sample Index')
    plt.ylabel('SPE Value')
    plt.legend()
    plt.title('SPE Statistic Over Time')
    plt.grid(True)
    plt.show()

# 设置字体以支持中文显示
#matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 加载训练集
train_data = np.loadtxt('../../TE_data/train_data/d01.dat')
test_data = np.loadtxt('../../TE_data/test_data/d01_te.dat')

test_data = test_data.T

# 对训练数据进行标准化 NMF不用标准化
#train_data = (train_data - np.mean(train_data, axis=0)) / np.std(train_data, axis=0)  # axis=0表示按列进行操作

# 计算出拉普拉斯矩阵
L = compute_laplacian_matrix(train_data, method='knn', gamma=0.5, n_neighbors=5)

# 使用 SJSNMF 方法返回 W 和 H
W, H = SJSNMF(train_data, L=L, rank=20, beta1=0.1, beta2=0.1, lam=1, s=15, max_iter=20, tol=1e-4)

# 计算重构矩阵 H_hat 和 X_hat
H_hat, X_hat = reconstruct_H_and_X(W, test_data)

# 计算 T2 和 SPE 统计量
T2, SPE = calculate_statistics(test_data, X_hat, H_hat)

# 绘制 T2 和 SPE 统计量的变化趋势
plot_statistics(T2, SPE)

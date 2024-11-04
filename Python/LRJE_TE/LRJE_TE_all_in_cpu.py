import numpy as np
import scipy.io
from sklearn.metrics import pairwise_distances
from scipy.sparse import csr_matrix
import matplotlib

# 设置字体以支持中文显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 加载数据
data = scipy.io.loadmat('TE_data.mat')
data = list(data.values())[3:] # 将数据转换为列表

# 分离训练数据和测试数据
testdata = data[:22]  # 前22个元素作为测试数据
train = np.array(data[22]) if len(data) > 22 else np.array(data[-1])  # 第23个元素作为训练数据，如果不存在则使用最后一个元素
#train = train.T  # 转置训练数据

# 计算训练数据的均值和标准差
train_mean = np.mean(train, axis=0)  # 按列计算均值
train_std = np.std(train, axis=0)  # 按列计算标准差
train_row, train_col = train.shape  # 获取训练数据的行数和列数
print("训练数据的维度为：", train_row, train_col)
#对数据进行标准化
X_train_std = (train - train_mean) / train_std

# 实现奇异值阈值化操作
def singular_value_thresholding(A, tau):
    """
    执行奇异值阈值化操作
    A: 输入矩阵
    tau: 阈值，这里是 1/mu
    """
    U, s, V = np.linalg.svd(A, full_matrices=False)
    s_thresholded = np.maximum(s - tau, 0)
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
    A = beta * X.T @ P @ P.T @ X + mu* X.T@X+mu * np.eye(n)  # 添加单位矩阵乘以mu以增加数值稳定性
    B = beta * X.T @ P @ P.T @ X + mu * X.T @ X - mu *X.T @ E + mu * Q + X.T @ Y1 - Y2

    # 解线性系统求解 Z
    # Z = np.linalg.solve(A, B)
    Z=(np.linalg.inv(A))@B
    print("Z 更新完成。")
    return Z

# 更新 E
def update_E(X, Z, Y1, mu, alpha):
    """
    根据提供的公式更新 E
    """
    m, n = X.shape
    V = X - X @ Z + Y1 / mu
    E = np.zeros_like(V)
    print("正在更新 E。")

    for i in range(n):
        norm_Vi = np.linalg.norm(V[:, i])
        if norm_Vi > alpha:
            E[:, i] = (norm_Vi - alpha) / norm_Vi * V[:, i]

    print("E 更新完成。")
    return E

# 计算权重矩阵 W
def compute_weight_matrix(X, t, k=7):
    # 计算点间欧式距离
    dists = pairwise_distances(X.T)
    # 创建k-近邻图
    np.fill_diagonal(dists, np.inf)
    neighbors = np.argsort(dists, axis=1)[:, :k]

    # 初始化权重矩阵
    W = np.zeros_like(dists)
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
    D = np.diag(W.sum(axis=1).A1)
    # 计算拉普拉斯矩阵 L
    L = D - W
    print("拉普拉斯矩阵 L 计算完成。")
    return L

# 更新 P
def update_P(X, Z, L, beta, gamma):
    """
    使用特征值分解来更新 P
    """
    n = X.shape[0]
    m = X.shape[1]
    I = np.eye(m)
    G = (Z - I) @ ((Z - I).T)
    A = X @ (beta * G + gamma * L) @ X.T

    # 计算特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eigh(A)

    # 选取最大的特征值对应的特征向量
    idx = np.argsort(eigenvalues)[::-1]  # 降序排列特征值
    P = eigenvectors[:, idx[:30]]  # 选取对应最大特征值的特征向量
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
    Z = Q = np.zeros((n, n))
    E= np.zeros((m, n))
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

# 计算协方并矩阵的逆，用于后续计算 T^2 统计量
Lambda = (Y_train @ Y_train.T) / (Y_train.shape[1] - 1)
Lambda_inv = np.linalg.inv(Lambda)
print("协方差矩阵及其逆计算完成。")


# 处理新样本
for i in range(22):
    test=np.array(testdata[i])
    n=test.shape[0] # 获取新样本的行数
    m=test.shape[1] # 获取新样本的列数

    #test= (test -train_mean) / train_std  # 对新数据进行标准化
    for j in range(n): # 对每一行进行处理
        y_new=P.T@test[j,:] # 投影到特征空间



import numpy as np
import scipy.io
from jupyterlab.semver import outside
from matplotlib import pyplot as plt
from networkx.classes import neighbors
from scipy.linalg import eigh
from scipy.spatial.distance import squareform
from sklearn.metrics import pairwise_distances
from scipy.sparse import csr_matrix
import matplotlib
from scipy.stats import gaussian_kde
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import kneighbors_graph
from sympy.abc import delta
from torch import pdist

#计算邻接矩阵 W_l
def calculate_W_l(delta,W_bar):
    #两个矩阵的内积
    W_l = delta*W_bar
    return W_l
def calculate_W_bar(X,t,k):
    # 计算点间欧式距离
    dists = pairwise_distances(X.T)
    # 创建k-近邻图
    np.fill_diagonal(dists, np.inf)

    neighbors = np.argsort(dists, axis=1)[:, :k]  # 选择前k个最近邻

    # 初始化权重矩阵
    W_bar = np.zeros_like(dists)
    for i in range(X.T.shape[0]):
        for j in neighbors[i]:
            W_bar[i, j] = np.exp(-dists[i, j] ** 2 / t)
            W_bar[j, i] = W_bar[i, j]  # 确保W是对称的
    return csr_matrix(W_bar)
def calculate_delta(E_d_i_ex_j, E_d_i_in_j):
    delta= E_d_i_ex_j / E_d_i_in_j
    return delta
def calculate_E_d_i_ex_j(X,k):
    """
       计算所有样本对于其非邻域样本的样本距离熵。

       参数:
       X : np.array
           数据集，每行代表一个样本。
       k : int
           考虑的邻居数量。

       返回:
       sdes : np.array
           每个样本的样本距离熵数组。
       """
    n_samples = X.shape[0]
    sde_E_d_i_ex_j = np.zeros(n_samples)

    # 计算点间欧氏距离
    dists = pairwise_distances(X)
    # 为自身距离设置无穷大以避免自身成为最近邻
    np.fill_diagonal(dists, np.inf)

    # 选择前k个最近邻
    neighbors = np.argsort(dists, axis=1)[:, :k]

    for i in range(n_samples):
        # 获取非邻居索引
        non_neighbors = [j for j in range(n_samples) if j not in neighbors[i]]

        # 如果没有非邻域样本，跳过
        if not non_neighbors:
            continue

        # 计算非邻居的距离概率
        p_ex = np.exp(-dists[i, non_neighbors])
        p_ex /= np.sum(p_ex)  # 归一化概率

        # 避免对数中的0值
        p_ex= np.clip(p_ex, 1e-10, None)

        # 计算样本距离熵
        sde_E_d_i_ex_j[i] = -np.sum(p_ex * np.log(p_ex))

    return sde_E_d_i_ex_j



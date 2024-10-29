import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import f, norm
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
train = train.T  # 转置训练数据

# 计算训练数据的均值和标准差
train_mean = np.mean(train, axis=0)  # 按列计算均值
train_std = np.std(train, axis=0)  # 按列计算标准差
train_row, train_col = train.shape  # 获取训练数据的行数和列数

# 对训练数据进行标准化
train = (train - train_mean) / train_std

# 计算协方差矩阵
sigmatrain = np.cov(train, rowvar=False)

# 对协方差矩阵进行特征值分解
lamda, T = np.linalg.eigh(sigmatrain)

# 将特征值和特征向量按降序排列
idx = np.argsort(lamda)[::-1]  # 获取特征值从大到小的索引
lamda = lamda[idx]  # 按降序排列特征值
T = T[:, idx]  # 按相应顺序排列特征向量

# 确定需要保留90%方差的主成分数量
num_pc = 1
while np.sum(lamda[:num_pc]) / np.sum(lamda) < 0.9:
    num_pc += 1

# 选择主成分
P = T[:, :num_pc]

# 计算99%和95%置信度下的T2控制限
T2UCL1 = num_pc * (train_row - 1) * (train_row + 1) * f.ppf(0.99, num_pc, train_row - num_pc) / (train_row * (train_row - num_pc))
T2UCL2 = num_pc * (train_row - 1) * (train_row + 1) * f.ppf(0.95, num_pc, train_row - num_pc) / (train_row * (train_row - num_pc))

# 计算99%置信度下的SPE控制限
theta = [np.sum(lamda[num_pc:]**i) for i in range(1, 4)]  # 计算theta1、theta2、theta3
h0 = 1 - 2 * theta[0] * theta[2] / (3 * theta[1]**2)  # 计算h0
ca = norm.ppf(0.99)  # 99%置信度下的标准正态分位数
SPE = theta[0] * (h0 * ca * np.sqrt(2 * theta[1]) / theta[0] + 1 + theta[1] * h0 * (h0 - 1) / theta[0]**2)**(1 / h0)

# 对测试数据集进行在线检测
for k in range(22):
    test = np.array(testdata[k])  # 获取第k组测试数据
    n = test.shape[0]  # 获取测试数据的行数

    # 对测试数据进行标准化
    test = (test - train_mean) / train_std

    I = np.eye(P.shape[0])  # 单位矩阵
    T2_test = np.zeros(n)  # 初始化T2统计量数组
    SPE_test = np.zeros(n)  # 初始化SPE统计量数组

    for i in range(n):
        # 计算第i个样本的T2统计量
        T2_test[i] = test[i, :] @ P @ np.linalg.inv(np.diag(lamda[:num_pc])) @ P.T @ test[i, :].T
        # 计算第i个样本的SPE统计量
        SPE_test[i] = test[i, :] @ (I - P @ P.T) @ test[i, :].T

    # 绘制图形
    plt.figure(k)
    plt.subplot(2, 1, 1)
    plt.plot(range(1, n + 1), T2_test, 'k')
    plt.title('主成分分析统计量T2')
    plt.xlabel('采样数')
    plt.ylabel('T^2')
    plt.axhline(y=T2UCL1, color='r', linestyle='--')  # 99%控制限
    plt.axhline(y=T2UCL2, color='g', linestyle='--')  # 95%控制限

    plt.subplot(2, 1, 2)
    plt.plot(range(1, n + 1), SPE_test, 'k')
    plt.title('主成分分析统计量SPE')
    plt.xlabel('采样数')
    plt.ylabel('SPE')
    plt.axhline(y=SPE, color='r', linestyle='--')  # 99%控制限

plt.show()

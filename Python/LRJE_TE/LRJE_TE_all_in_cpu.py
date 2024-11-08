import numpy as np
import scipy.io
from matplotlib import pyplot as plt
from sklearn.metrics import pairwise_distances
from scipy.sparse import csr_matrix
import matplotlib
from scipy.stats import gaussian_kde
import func
# 设置字体以支持中文显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 加载数据
data = scipy.io.loadmat('TE_data.mat')
data = list(data.values())[3:] # 将数据转换为列表

# 分离训练数据和测试数据
train = np.array(data[22]) if len(data) > 22 else np.array(data[-1])  # 第23个元素作为训练数据，如果不存在则使用最后一个元素
testdata = data[:22]  # 前22个元素作为测试数据

# 计算训练数据的均值和标准差
train_mean = np.mean(train, axis=1)  # 按行计算均值
train_std = np.std(train, axis=1)  # 按行计算标准差
train_row, train_col = train.shape  # 获取训练数据的行数和列数
print("训练数据的维度为：", train_row, train_col)
#对数据进行标准化
X_train_std = (train - train_mean[:, np.newaxis]) / train_std[:, np.newaxis]


# 使用 LRJE 方法计算 P
print("开始使用 LRJE 方法计算 P。")
P = func.lrje(X_train_std, alpha=0.5, beta=0.5, gamma=0.5)  # 计算投影矩阵 P
print("P 计算完成。")
Y_train = P.T @ X_train_std  # 投影到特征空间
print("训练数据投影到特征空间完成。")

# 计算协方并矩阵的逆，用于后续计算 T^2 统计量
Lambda = (Y_train @ Y_train.T) / (Y_train.shape[1] - 1)
Lambda_inv = np.linalg.inv(Lambda)
print("协方差矩阵及其逆计算完成。")

# 计算训练数据的 T² 和 SPE 统计量
n_train = X_train_std.shape[1]
T2_train = np.zeros(n_train)
SPE_train = np.zeros(n_train)
I = np.eye(X_train_std.shape[0])

for j in range(n_train):
    y_train_new = P.T @ X_train_std[:, j]  # 投影到特征空间
    T2_train[j] = y_train_new @ Lambda_inv @ y_train_new.T  # 计算 T² 统计量
    x_residual = (I - P @ P.T) @ X_train_std[:, j]  # 计算残差空间的投影
    SPE_train[j] = np.sum(np.square(x_residual))  # 计算 SPE 统计量

# 使用核密度估计计算 T² 和 SPE 的控制限
alpha = 0.99  # 置信水平

# T² 控制限
kde_T2 = gaussian_kde(T2_train)
T2_limit = np.percentile(T2_train, alpha * 100)

# SPE 控制限
kde_SPE = gaussian_kde(SPE_train)
SPE_limit = np.percentile(SPE_train, alpha * 100)

# 处理新样本
#先计算两个测试数据
testnum=22
for i in range(testnum):
    test=np.array(testdata[i])
    n=test.shape[0] # 获取新样本的行数
    m=test.shape[1] # 获取新样本的列数

    #test= (test -train_mean) / train_std  # 对新数据进行标准化
    train_std = np.where(train_std == 0, 1, train_std)# 确保标准差非零
    test = (test - train_mean.reshape(1, -1)) / train_std.reshape(1, -1)

    T2_test = np.zeros(n)  # 初始化T2统计量数组
    SPE_test = np.zeros(n)  # 初始化SPE统计量数组
    I = np.eye(m)

    for j in range(n): # 对每一行进行处理
        y_new=P.T@test[j,:] # 投影到特征空间
        T2_test[j] = y_new @ Lambda_inv @ y_new.T  # 计算 T² 统计量

        x_res = (I - P@P.T) @ test[j, :]  # 计算残差空间的投影
        SPE_test[j] = np.sum(np.square(x_res))  # 计算 SPE 统计量

    # 绘制 T² 和 SPE 统计量的折线图
    plt.figure(figsize=(14, 6))

    # 绘制 T² 统计量图
    plt.subplot(1, 2, 1)
    plt.plot(T2_test, label='T2 Statistic')
    plt.axhline(y=T2_limit, color='r', linestyle='--', label=f'T2 Control Limit ({alpha * 100}%)')
    plt.title(f'T2 Statistic for Test Set {i + 1}')
    plt.xlabel('Sample index')
    plt.ylabel('T2 Value')
    plt.grid(True)
    plt.legend()

    # 绘制 SPE 统计量图
    plt.subplot(1, 2, 2)
    plt.plot(SPE_test, label='SPE Statistic')
    plt.axhline(y=SPE_limit, color='r', linestyle='--', label=f'SPE Control Limit ({alpha * 100}%)')
    plt.title(f'SPE Statistic for Test Set {i + 1}')
    plt.xlabel('Sample index')
    plt.ylabel('SPE Value')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

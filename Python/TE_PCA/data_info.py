import scipy.io

# 加载 .mat 文件
data = scipy.io.loadmat('TE_data.mat')

# 打印数据中的键，以查看所有变量的名称
print(data.keys())

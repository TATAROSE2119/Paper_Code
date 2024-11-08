import numpy as np
import scipy.io
from matplotlib import pyplot as plt
from sklearn.metrics import pairwise_distances
from scipy.sparse import csr_matrix
import matplotlib
from scipy.stats import gaussian_kde
from func import *

# 设置字体以支持中文显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
#加载训练集
train_data=np.loadtxt('train_data/d01.dat')
test_data=np.loadtxt('test_data/d01_te.dat')
#对训练数据进行标准化
train_data=(train_data-np.mean(train_data,axis=0))/np.std(train_data,axis=0) # axis=0表示按列进行操作
#计算出拉普拉斯矩阵
L=reduce_laplacian_matrix(compute_laplacian_matrix(train_data,method='rbf',gamma=0.5,n_neighbors=5),rank=20)
#使用SJSNMF方法返回W，H
W,H=SJSNMF(train_data,L=L, rank=20, beta1=0.1, beta2=0.1,lam=1,s=0.5,max_iter=100,tol=1e-4)
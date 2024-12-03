import numpy as np
import scipy.io
from matplotlib import pyplot as plt
from scipy.linalg import eigh
from scipy.spatial.distance import squareform
from sklearn.metrics import pairwise_distances
from scipy.sparse import csr_matrix
import matplotlib
from scipy.stats import gaussian_kde
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import kneighbors_graph
from torch import pdist

# Load train data
train_data = np.loadtxt('../../TE_data/train_data/d01.dat')
# Load test data
test_data = np.loadtxt('../../TE_data/test_data/d01_te.dat')

#print the shape of train_data
print(train_data.shape)
#进行数据的标准化
train_data = (train_data - np.mean(train_data, axis=0)) / np.std(train_data, axis=0) # axis=0表示按列进行操作
test_data = (test_data - np.mean(test_data, axis=0)) / np.std(test_data, axis=0) # axis=0表示按列进行操作



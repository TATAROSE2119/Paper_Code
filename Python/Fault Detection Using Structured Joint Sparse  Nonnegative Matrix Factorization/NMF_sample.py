import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import NMF

# 创建数据矩阵
data = np.array([
    [85, 105, 7.1, 0.5],
    [86, 106, 7.0, 0.55],
    [87, 107, 7.0, 0.58],
    [88, 108, 7.1, 0.60],
    [89, 109, 7.1, 500],
    [90, 110, 7.2, 0.65],
    [91, 2110, 7.2, 0.68],
    [92, 112, 7.3, 0.70],
    [93, 113, 7.3, 0.73],
    [94, 114, 7.4, 0.75]
])

# 初始化NMF模型
model = NMF(n_components=2, init='random', random_state=0)
W = model.fit_transform(data)
H = model.components_

# 重构数据
reconstructed = np.dot(W, H)

# 可视化原始数据和重构数据
plt.figure(figsize=(10, 6))
plt.subplot(121)
plt.title('Original Data')
plt.imshow(data, aspect='auto', interpolation='none',cmap='summer')
plt.colorbar()
plt.xlabel('Features')
plt.ylabel('Samples')
plt.xticks(range(data.shape[1]), ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4'])
plt.yticks(range(data.shape[0]), ['Sample 1', 'Sample 2', 'Sample 3', 'Sample 4', 'Sample 5', 'Sample 6', 'Sample 7', 'Sample 8', 'Sample 9', 'Sample 10'])

plt.subplot(122)
plt.title('Reconstructed Data')
plt.imshow(reconstructed, aspect='auto', interpolation='none',cmap='summer')
plt.colorbar()
plt.xlabel('Features')
plt.ylabel('Samples')
plt.xticks(range(data.shape[1]), ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4'])
plt.yticks(range(data.shape[0]), ['Sample 1', 'Sample 2', 'Sample 3', 'Sample 4', 'Sample 5', 'Sample 6', 'Sample 7', 'Sample 8', 'Sample 9', 'Sample 10'])

plt.tight_layout()
plt.show()

# 可视化W矩阵
plt.figure(figsize=(8, 4))
plt.title('W Matrix (Mixing Matrix)')
plt.imshow(W, aspect='auto', interpolation='none', cmap='summer')
plt.colorbar()
plt.xlabel('Components')
plt.ylabel('Samples')
plt.xticks(range(W.shape[1]), ['Component 1', 'Component 2'])
plt.yticks(range(W.shape[0]), ['Sample 1', 'Sample 2', 'Sample 3', 'Sample 4', 'Sample 5', 'Sample 6', 'Sample 7', 'Sample 8', 'Sample 9', 'Sample 10'])
plt.show()

# 可视化H矩阵
plt.figure(figsize=(8, 4))
plt.title('H Matrix (Feature Matrix)')
plt.imshow(H, aspect='auto', interpolation='none', cmap='summer')
plt.colorbar()
plt.xlabel('Features')
plt.ylabel('Components')
plt.xticks(range(H.shape[1]), ['Feature 1', 'Feature 2', 'Feature 3', 'Feature 4'])
plt.yticks(range(H.shape[0]), ['Component 1', 'Component 2'])
plt.show()

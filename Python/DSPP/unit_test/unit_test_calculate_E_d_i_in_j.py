import unittest
import numpy as np
from Python.DSPP.DSPP_func import calculate_E_d_i_in_j

class TestCalculateE_d_i_in_j(unittest.TestCase):

    def test_simple_case(self):  # 测试一个简单的例子
        X = np.array([[0, 0], [1, 1], [2, 2]])
        k = 2
        expected_entropy = np.array([0.0, 0.0, 0.0])
        entropy = calculate_E_d_i_in_j(X, k)
        np.testing.assert_array_almost_equal(entropy, expected_entropy, decimal=6)

    def test_single_point(self):
        X = np.array([[0, 0]])
        k = 1
        expected_entropy = np.array([0.0])
        entropy = calculate_E_d_i_in_j(X, k)
        np.testing.assert_array_almost_equal(entropy, expected_entropy, decimal=6)

    def test_large_k(self):
        X = np.array([[0, 8], [1, 10], [2, 6], [3, 4]])
        k = 3
        expected_entropy = np.array([0.790523, 0.458943, 0.893223, 0.301628])
        entropy = calculate_E_d_i_in_j(X, k)
        np.testing.assert_array_almost_equal(entropy, expected_entropy, decimal=6)  # 精度

    def test_large_dataset(self):
        """
        测试具有 480 个样本和 52 个特征的数据集
        """
        np.random.seed(42)  # 设置随机种子以确保结果可复现
        n_samples = 480
        n_features = 52
        X = np.random.rand(n_samples, n_features)
        k = 10  # 选择一个合理的 k 值
        entropy = calculate_E_d_i_in_j(X, k)
        self.assertEqual(entropy.shape, (n_samples,))  # 检查返回的熵值数组的形状是否正确
        # 由于随机数据的熵值难以预测，这里不进行具体的数值比较，而是检查返回值的形状和类型
        self.assertTrue(np.all(entropy >= 0))  # 检查所有熵值是否非负

if __name__ == "__main__":
    unittest.main()

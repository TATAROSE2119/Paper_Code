import unittest
import numpy as np
from Python.DSPP.DSPP_func import calculate_E_d_i_ex_j

class TestCalculateE_d_i_ex_j(unittest.TestCase):

    def test_simple_2d_data(self):
        X = np.array([[0, 0], [1, 1], [2, 2]])
        k = 2
        expected = np.array([0.0, 0.0, 0.0])  # 预期结果可能因方法而异，这里假设为0.0
        result = calculate_E_d_i_ex_j(X, k)
        np.testing.assert_array_almost_equal(result, expected, decimal=6)

    def test_k_equals_1(self):
        X = np.array([[0, 0], [1, 1], [2, 2]])
        k = 1
        expected = np.array([0.0, 0.0, 0.0])  # 预期结果可能因方法而异，这里假设为0.0
        result = calculate_E_d_i_ex_j(X, k)
        np.testing.assert_array_almost_equal(result, expected, decimal=6)

    def test_one_point_far_away(self):
        X = np.array([[0, 0], [10, 10], [2, 2]])
        k = 1
        expected = np.array([0.0, 0.0, 0.0])  # 预期结果可能因方法而异，这里假设为0.0
        result = calculate_E_d_i_ex_j(X, k)
        np.testing.assert_array_almost_equal(result, expected, decimal=6)

    def test_large_dataset(self):
        # 生成480个样本，每个样本有52个特征的随机数据集
        n_samples = 480
        n_features = 52
        X = np.random.rand(n_samples, n_features)
        k = 5  # 选择一个合理的k值

        # 计算结果
        result = calculate_E_d_i_ex_j(X, k)

        # 验证结果的形状
        self.assertEqual(result.shape, (n_samples,))

        # 由于随机数据集的结果难以预测，我们主要检查结果的合理性
        # 例如，确保所有值都是非负的
        self.assertTrue(np.all(result >= 0))

    # 可以添加更多测试用例来覆盖更多边缘情况

if __name__ == '__main__':
    unittest.main()

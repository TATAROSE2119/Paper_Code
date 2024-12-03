import unittest
import numpy as np
from scipy.sparse import csr_matrix
from Python.DSPP.DSPP_func import calculate_W_bar  # 确保路径正确，以便导入calculate_W_bar

class TestCalculateWBar(unittest.TestCase):
    def setUp(self):
        # 设置测试数据
        self.X = np.array([[0, 1], [1, 0], [2, 3]])
        self.t = 1.0
        self.k = 2

    def test_calculate_W_bar_type(self):
        # 测试返回值类型
        W_bar = calculate_W_bar(self.X, self.t, self.k)
        self.assertIsInstance(W_bar, csr_matrix, "The result should be a csr_matrix")

    def test_calculate_W_bar_symmetry(self):
        # 测试矩阵是否对称
        W_bar = calculate_W_bar(self.X, self.t, self.k)
        self.assertTrue((W_bar != W_bar.T).nnz == 0, "The matrix should be symmetric")

    def test_calculate_W_bar_diagonal(self):
        # 测试对角线是否为零
        W_bar = calculate_W_bar(self.X, self.t, self.k)
        self.assertTrue(np.all(W_bar.diagonal() == 0), "The diagonal of the matrix should be zero")

    def test_calculate_W_bar_nonzero_elements(self):
        # 测试非零元素的值
        W_bar = calculate_W_bar(self.X, self.t, self.k)
        expected_nonzero_elements = np.array([np.exp(-1.0), np.exp(-5.0), np.exp(-1.0)])
        actual_nonzero_elements = W_bar.data
        np.testing.assert_array_almost_equal(actual_nonzero_elements, expected_nonzero_elements, decimal=6)

if __name__ == '__main__':
    unittest.main()

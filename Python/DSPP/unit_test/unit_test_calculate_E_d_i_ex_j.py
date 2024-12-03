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

    # 可以添加更多测试用例来覆盖更多边缘情况

if __name__ == '__main__':
    unittest.main()

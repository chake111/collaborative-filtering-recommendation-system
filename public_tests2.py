"""
public_tests2.py
基于深度学习的内容过滤推荐系统测试函数
"""

import numpy as np


def test_sq_dist(sq_dist_func):
    """
    测试平方距离函数

    参数:
        sq_dist_func: 待测试的平方距离函数
    """
    print("Testing sq_dist function...")

    # 测试用例 1: 相同向量，距离应为0
    a1 = np.array([1.0, 2.0, 3.0])
    b1 = np.array([1.0, 2.0, 3.0])
    result1 = sq_dist_func(a1, b1)
    expected1 = 0.0
    assert np.isclose(result1, expected1), f"Test 1 failed: expected {expected1}, got {result1}"
    print(f"  Test 1 passed: sq_dist([1,2,3], [1,2,3]) = {result1}")

    # 测试用例 2: 稍有不同的向量
    a2 = np.array([1.1, 2.1, 3.1])
    b2 = np.array([1.0, 2.0, 3.0])
    result2 = sq_dist_func(a2, b2)
    expected2 = 0.03  # (0.1)^2 + (0.1)^2 + (0.1)^2 = 0.03
    assert np.isclose(result2, expected2, atol=1e-6), f"Test 2 failed: expected {expected2}, got {result2}"
    print(f"  Test 2 passed: sq_dist([1.1,2.1,3.1], [1,2,3]) = {result2:.4f}")

    # 测试用例 3: 正交向量
    a3 = np.array([0, 1, 0])
    b3 = np.array([1, 0, 0])
    result3 = sq_dist_func(a3, b3)
    expected3 = 2.0  # (0-1)^2 + (1-0)^2 + (0-0)^2 = 2
    assert np.isclose(result3, expected3), f"Test 3 failed: expected {expected3}, got {result3}"
    print(f"  Test 3 passed: sq_dist([0,1,0], [1,0,0]) = {result3}")

    # 测试用例 4: 更大的向量
    a4 = np.array([1, 2, 3, 4, 5])
    b4 = np.array([5, 4, 3, 2, 1])
    result4 = sq_dist_func(a4, b4)
    expected4 = 40.0  # (1-5)^2 + (2-4)^2 + (3-3)^2 + (4-2)^2 + (5-1)^2 = 16+4+0+4+16 = 40
    assert np.isclose(result4, expected4), f"Test 4 failed: expected {expected4}, got {result4}"
    print(f"  Test 4 passed: sq_dist([1,2,3,4,5], [5,4,3,2,1]) = {result4}")

    # 测试用例 5: 浮点数向量
    a5 = np.array([0.5, 1.5, 2.5])
    b5 = np.array([1.5, 2.5, 3.5])
    result5 = sq_dist_func(a5, b5)
    expected5 = 3.0  # (0.5-1.5)^2 + (1.5-2.5)^2 + (2.5-3.5)^2 = 1+1+1 = 3
    assert np.isclose(result5, expected5), f"Test 5 failed: expected {expected5}, got {result5}"
    print(f"  Test 5 passed: sq_dist([0.5,1.5,2.5], [1.5,2.5,3.5]) = {result5}")

    print("\n\033[92mAll tests passed!\033[0m")
    return True


def test_nn_model(model, num_user_features, num_item_features):
    """
    测试神经网络模型结构

    参数:
        model: Keras模型
        num_user_features: 用户特征数量
        num_item_features: 项目特征数量
    """
    print("Testing neural network model structure...")

    # 检查模型输入
    assert len(model.inputs) == 2, "Model should have 2 inputs (user and item)"
    print("  ✓ Model has correct number of inputs")

    # 检查模型输出
    assert len(model.outputs) == 1, "Model should have 1 output"
    print("  ✓ Model has correct number of outputs")

    print("\n\033[92mModel structure tests passed!\033[0m")
    return True


# 导出测试函数
__all__ = ['test_sq_dist', 'test_nn_model']

import numpy as np

def test_cofi_cost_func(target_func):
    """
    测试协同过滤代价函数 cofi_cost_func
    """
    np.random.seed(1)
    num_users    = 4
    num_movies   = 5
    num_features = 3

    X = np.random.randn(num_movies,   num_features)
    W = np.random.randn(num_users,    num_features)
    b = np.random.randn(1,            num_users)
    Y = np.random.randn(num_movies,   num_users)
    R = (np.random.rand(num_movies,   num_users) > 0.5).astype(float)

    # ---- 无正则化测试 ----
    cost_no_reg = target_func(X, W, b, Y, R, lambda_=0)
    assert cost_no_reg > 0, "代价值应为正数"
    print(f"[✓] 无正则化代价: {cost_no_reg:.4f}")

    # ---- 有正则化测试（代价应更大）----
    cost_reg = target_func(X, W, b, Y, R, lambda_=1.5)
    assert cost_reg >= cost_no_reg, "正则化后代价应 >= 无正则化代价"
    print(f"[✓] 有正则化代价: {cost_reg:.4f}")

    # ---- 形状/类型检查 ----
    assert np.isscalar(cost_no_reg) or cost_no_reg.shape == (), \
        "返回值应为标量"
    print("[✓] 返回值类型正确（标量）")

    print("\n所有测试通过！cofi_cost_func 实现正确 ✅")

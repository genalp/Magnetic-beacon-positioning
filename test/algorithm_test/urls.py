import numpy as np
from scipy.optimize import least_squares

# 定义非线性观测模型
def hx(state, xi, yi, zi):
    """ 计算理论磁场振幅 """
    K = 300000.0
    x0, y0, z0 = state
    x = x0 - xi
    y = y0 - yi
    z = z0 - zi
    r = np.sqrt(x*x + y*y + z*z)
    B = K * np.sqrt(3*z*z + r*r) / (r*r*r*r)
    return B

# 生成模拟测量数据
def generate_measurement(true_state, noise_std=10):
    """ 生成单个测量数据 """
    xi, yi, zi = np.random.rand(3) * 10
    B = hx(true_state, xi, yi, zi) + np.random.normal(0, noise_std)
    return B, xi, yi, zi

# NRLS 更新函数
def update_state_nl_least_squares(measurements, initial_guess):
    """ 使用非线性最小二乘法更新状态 """
    def residual(state, measurements):
        return [hx(state, xi, yi, zi) - B for B, xi, yi, zi in measurements]

    result = least_squares(residual, initial_guess, args=(measurements,))
    return result.x

# 主程序
def main():
    true_state = np.array([1, 2, 3])  # 真实状态（用于生成模拟数据）
    measurements = []
    initial_guess = np.array([0, 0, 0])  # 初始状态估计

    # 模拟一系列测量并更新状态估计
    for _ in range(100):
        new_measurement = generate_measurement(true_state)
        measurements.append(new_measurement)
        estimated_state = update_state_nl_least_squares(measurements, initial_guess)
        initial_guess = estimated_state  # 更新初始猜测

    print("Estimated State:", estimated_state)
    print("True State:", true_state)

main()

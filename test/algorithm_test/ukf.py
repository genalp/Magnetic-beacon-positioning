import numpy as np
from scipy.optimize import least_squares
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints
import matplotlib.pyplot as plt

# 定义状态转移和观测函数
def fx(state, dt):
    return state  # 状态转移函数 (静态系统)

def hx(state, xi, yi, zi):
    K = 300000.0
    x0, y0, z0 = state
    x = x0 - xi
    y = y0 - yi
    z = z0 - zi
    r = np.sqrt(x*x + y*y + z*z)
    B = K * np.sqrt(3*z*z + r*r) / (r*r*r*r)
    return np.array([B])

# 生成模拟测量数据
def generate_measurements(true_state, num_points=100):
    measurements = []
    for _ in range(num_points):
        xi, yi, zi = np.random.rand(3) * 10
        B = hx(true_state, xi, yi, zi)[0] + np.random.normal(0, 10)
        measurements.append((B, xi, yi, zi))
    return measurements

# 使用非线性最小二乘法初始化状态
def initialize_state_nl_least_squares(measurements, initial_guess):
    def residual(state, measurements):
        residuals = []
        for B, xi, yi, zi in measurements:
            predicted_B = hx(state, xi, yi, zi)[0]
            residuals.append(predicted_B - B)
        return residuals

    result = least_squares(residual, initial_guess, args=(measurements,))
    return result.x

# 主程序
def main(use_nl_least_squares_init=True):
    true_state = np.array([1, 2, 3])  # 真实状态 (用于生成模拟数据)
    measurements = generate_measurements(true_state)

    points = MerweScaledSigmaPoints(n=3, alpha=0.1, beta=2., kappa=-1)
    ukf = UKF(dim_x=3, dim_z=1, fx=fx, hx=hx, dt=1, points=points)
    # ukf.P *= 100
    ukf.R = np.array([[10]])
    ukf.Q = np.eye(3) * 0.1

    if use_nl_least_squares_init:
        initial_state = initialize_state_nl_least_squares(measurements[:5], [0, 0, 0])
        print(initial_state)
        ukf.x = initial_state

    ukf_state_estimations = []
    for B, xi, yi, zi in measurements:
        ukf.predict()
        ukf.update(B, xi=xi, yi=yi, zi=zi)
        ukf_state_estimations.append(ukf.x.copy())

    ukf_state_estimations = np.array(ukf_state_estimations)

    # 输出最终的估计结果
    ukf_estimated_state = ukf_state_estimations[-1]
    ukf_error = np.linalg.norm(true_state - ukf_estimated_state)
    print("UKF Estimated State:", ukf_estimated_state)
    print("UKF Error:", ukf_error)

    plt.figure(figsize=(12, 6))

    for i, label in enumerate(['x', 'y', 'z']):
        plt.subplot(1, 3, i + 1)
        plt.plot(ukf_state_estimations[:, i], label=f'UKF Estimated {label}')
        plt.axhline(y=true_state[i], color='r', linestyle='--', label=f'True {label}')
        plt.xlabel('Measurement Number')
        plt.ylabel('State Value')
        plt.title(f'UKF Convergence of {label}')
        plt.legend()

    plt.tight_layout()
    plt.show()
    

main(use_nl_least_squares_init=False)

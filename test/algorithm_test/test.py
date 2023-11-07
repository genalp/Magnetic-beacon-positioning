from magnetic_beacon import magnetic_beacon
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
from sko.PSO import PSO
from sko.SA import SA
import pandas as pd

beacons = [
    [[0, 2, 0], 4]
    ,[[1, 0, 0], 5]
    ,[[5, 0, 0], 6]
    ,[[3, 0, 0], 8]
    # ,[[1, 1, 0], 10]
]

B = []

m = magnetic_beacon(
    sensor_position = [8,8,17], 
    if_add_noise=True,
    if_add_angle_error=True,
    max_angle_error=1,
    if_add_beacon_error=True
    )
for beacon in beacons:
    m.add_beacon(beacon[0], beacon[1])
m.run()
# m.draw_signal()


# 定义IIR低通滤波器的参数
cutoff_frequency = 30  # 截止频率 (Hz)
filter_order = 4  # 滤波器阶数
# 创建IIR低通滤波器
b, a = signal.butter(filter_order, cutoff_frequency / (0.5 * m.sensor_fs), btype='low')
filtered_signal = signal.lfilter(b, a, m.signal_output)


# 进行FFT
fft_result = np.fft.fft(filtered_signal)
fft_freq = np.fft.fftfreq(m.signal_len, 1 / m.sensor_fs)
# 绘制FFT结果
# plt.figure(figsize=(10, 4))
# plt.plot(fft_freq, np.abs(fft_result))
# plt.title('FFT Result')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Amplitude')
# plt.xlim(0, 50)  # 限制绘图范围，根据需要调整


# 处理FFT结果
for beacon in beacons:
    # 提取特定频率对应的幅值
    target_frequency = beacon[1]
    # 计算目标频率在FFT结果中的索引
    index = int(target_frequency * len(fft_result) / m.sensor_fs)
    # 根据FFT的幅值公式将幅值除以FFT数据点数量的一半
    amplitude_at_target_frequency = np.abs(fft_result[index]) / (len(fft_result) / 2)
    B.append(amplitude_at_target_frequency)

# print(B)



# plt.figure(figsize=(10, 8))

# plt.subplot(2, 1, 1)
# plt.plot(m.signal_output, label='Original Signal')
# plt.title('Original Signal')

# plt.subplot(2, 1, 2)
# plt.plot(filtered_signal, label='Noisy Signal', color='red')
# plt.title('Signal with Gaussian Noise')

# plt.show()




def obj_fun(p):
    global m, beacons, B

    K = m.M
    x, y, z = p
    # x, y = p
    # z = -3

    fn = []
    f = 0

    # 优化模型
    for i in range(len(beacons)):
        fi = np.log(B[i]) - np.log(K) - \
            0.5 * np.log(3*(beacons[i][0][2]-z)**2 + (beacons[i][0][0]-x)**2 + (beacons[i][0][1]-y)**2 + (beacons[i][0][2]-z)**2) + \
            2 * np.log((beacons[i][0][0]-x)**2 + (beacons[i][0][1]-y)**2 + (beacons[i][0][2]-z)**2)
        fn.append(fi)
    
    for fi in fn:
        fi = fi * 1e3
        f += fi**2
    if f == 0:
        f = 1e-100
    # f = np.log10(f)
    return f

# pso = PSO(func=obj_fun, n_dim=3, 
#           pop=50, max_iter=500, 
#           lb=[0, 0, 5], ub=[20, 20, 50], 
#           w=0.9, c1=0.5, c2=0.5)
# pso.run()
# print(' x :', pso.gbest_x[0], ' y :', pso.gbest_x[1], ' z: ', pso.gbest_x[2], 'best_y is', pso.gbest_y)
# plt.plot(pso.gbest_y_hist)
# plt.show()

sa = SA(func=obj_fun, x0=[0, 0, 5], T_max=1, T_min=1e-9, L=300, max_stay_counter=150)
best_x, best_y = sa.run()
print('best_x:', best_x, 'best_y', best_y)
# plt.plot(pd.DataFrame(sa.best_y_history).cummin(axis=0))
# plt.show()
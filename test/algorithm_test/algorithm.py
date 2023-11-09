from magnetic_beacon import magnetic_beacon
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
from sko.PSO import PSO
from sko.SA import SA
import pandas as pd
from scipy.stats import gaussian_kde
import threading

beacons = [
    [[0, 0, 0], 3]
    ,[[1, 0, 1], 4]
    ,[[2, 0, 0], 5]
    ,[[0, 1, 1], 6]
    ,[[0, 2, 0], 8]
    ,[[2, 2, 2], 10]
]

B = []
sensor_position = [3,7,1]
beacon_list = []
m = magnetic_beacon()
data = []

# 1: 经典算法
# 2: 简单质心
# 3: 加权质心
algorithm_num = 1
save_path = ['sorted_data8.txt', 'cdf8.txt']

# 初始化信标
def init_beacons():
    global m, beacons
    m = magnetic_beacon(
        sensor_position = sensor_position, 
        if_add_noise=True,
        if_add_angle_error=True,
        max_angle_error=0.3,
        if_add_beacon_error=False
        )
    for beacon in beacons:
        m.add_beacon(beacon[0], beacon[1])
    m.run()
    # m.draw_signal()

# IIR滤波
def beacon_IIR():
    global m
    # 定义IIR低通滤波器的参数
    cutoff_frequency = 40  # 截止频率 (Hz)
    filter_order = 4  # 滤波器阶数
    # 创建IIR低通滤波器
    b, a = signal.butter(filter_order, cutoff_frequency / (0.5 * m.sensor_fs), btype='low')
    filtered_signal = signal.lfilter(b, a, m.signal_output)
    return filtered_signal

# 进行FFT
def beacon_FFT(filtered_signal):
    global B, m
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


# 定位模型
def obj_fun(p):
    global m, beacons, B, beacon_list

    K = m.M
    x, y, z = p

    fn = []
    f = 0

    # 优化模型
    for i in beacon_list:
        # fi = np.log(B[i]) - np.log(K) - \
        #     0.5 * np.log(3*(beacons[i][0][2]-z)**2 + (beacons[i][0][0]-x)**2 + (beacons[i][0][1]-y)**2 + (beacons[i][0][2]-z)**2) + \
        #     2 * np.log((beacons[i][0][0]-x)**2 + (beacons[i][0][1]-y)**2 + (beacons[i][0][2]-z)**2)
        
        fi = K * np.sqrt(3*(beacons[i][0][2]-z)**2 + (beacons[i][0][0]-x)**2 + (beacons[i][0][1]-y)**2 + (beacons[i][0][2]-z)**2) - \
            B[i] * ((beacons[i][0][0]-x)**2 + (beacons[i][0][1]-y)**2 + (beacons[i][0][2]-z)**2)**2

        fn.append(fi)
    
    for fi in fn:
        fi = fi * 1e3
        f += fi**2
    # if f == 0:
    #     f = 1e-100
    # f = np.log10(f)
    return f

def SA_position():
    sa = SA(func=obj_fun, x0=[0, 0, 5], T_max=1, T_min=1e-9, L=300, max_stay_counter=150)
    best_x, best_y = sa.run()
    # print('best_x:', best_x, 'best_y', best_y)
    # plt.plot(pd.DataFrame(sa.best_y_history).cummin(axis=0))
    # plt.show()
    return best_x

def PSO_position():
    pso = PSO(func=obj_fun, n_dim=3, 
              pop=50, max_iter=500, 
              lb=[0, 0, 0], ub=[20, 20, 10], 
              w=0.9, c1=0.5, c2=0.5)
    pso.run()
    # print(' x :', pso.gbest_x[0], ' y :', pso.gbest_x[1], ' z: ', pso.gbest_x[2], 'best_y is', pso.gbest_y)
    # plt.plot(pso.gbest_y_hist)
    # plt.show()
    return pso.gbest_x

# init_beacons()
# filtered_signal = beacon_IIR()
# beacon_FFT(filtered_signal)

# 计算权重
def calculate_weights(n = 1):
    global B, beacon_list
    b = 0
    for i in beacon_list:
        b += B[i]
    b = np.power(b, n)
    return b

# 加权质心
def weighted_centroid(positions, weights):
    x = 0
    y = 0
    z = 0
    for i in range(len(positions)):
        x += positions[i][0] * weights[i]
        y += positions[i][1] * weights[i]
        z += positions[i][2] * weights[i]
    n = sum(weights)
    x /= n
    y /= n
    z /= n
    return [x, y, z]




def test():
    global sensor_position, m, algorithm_num, beacon_list, save_path
    data = []
    position = []

    for x in range(1,9):
        for y in range(1,9):
            for z in range(3,5):
                # 数值初始化
                n = 0
                B.clear()
                # 更改传感器位置，重新生成信号
                sensor_position = [x,y,z]
                m.set_sensor_position(sensor_position)
                m.run()
                # 滤波和FFT
                filtered_signal = beacon_IIR()
                beacon_FFT(filtered_signal)

                # 检查算法类型
                if algorithm_num == 1:
                    beacon_list.clear()
                    beacon_list = list(range(len(beacons)))
                    position = SA_position()
                    # position = PSO_position()
                elif algorithm_num == 2:
                    every_position = []
                    for i in range(len(beacons)-1):
                        beacon_list.clear()
                        beacon_list = list(range(len(beacons)))
                        beacon_list.remove(i)
                        position = SA_position()
                        every_position.append(position)
                    print(np.array(every_position))
                    position = np.mean(np.abs(every_position), axis=0)
                elif algorithm_num == 3:
                    every_position = []
                    weights = []
                    for i in range(len(beacons)-1):
                        beacon_list.clear()
                        beacon_list = list(range(len(beacons)))
                        beacon_list.remove(i)
                        position = SA_position()
                        every_position.append(position)
                        weights.append(calculate_weights(4))
                    print(np.array(every_position))
                    position = weighted_centroid(every_position, weights)

                print(position)
                for i in range(3):
                    n += (np.abs(position[i]) - sensor_position[i])**2
                n = np.sqrt(n)
                print(n)
                data.append(n)

    # for i in range(100):
    #     n = 0
    #     init_beacons()
    #     filtered_signal = beacon_IIR()
    #     beacon_FFT(filtered_signal)

    #     # print(m.beacon_position_noise)

    #     sa = SA(func=obj_fun, x0=[0, 0, 5], T_max=1, T_min=1e-9, L=300, max_stay_counter=150)
    #     best_x, best_y = sa.run()
    #     print('best_x:', best_x, 'best_y', best_y)

    #     # pso = PSO(func=obj_fun, n_dim=3, 
    #     #           pop=50, max_iter=500, 
    #     #           lb=[0, 0, 5], ub=[20, 20, 50], 
    #     #           w=0.9, c1=0.5, c2=0.5)
    #     # pso.run()
    #     # print(' x :', pso.gbest_x[0], ' y :', pso.gbest_x[1], ' z: ', pso.gbest_x[2], 'best_y is', pso.gbest_y)


    #     for i in range(3):
    #         n += (np.abs(best_x[i]) - sensor_position[i])**2
    #         # n += (np.abs(pso.gbest_x[i]) - sensor_position[i])**2
    #     n = np.sqrt(n)
    #     print(n)
    #     data.append(n)
    #     # print(i)
    # 计算CDF
    sorted_data = np.sort(data)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

    np.savetxt(save_path[0],sorted_data)
    np.savetxt(save_path[1],cdf)

    # 绘制CDF函数图
    plt.plot(sorted_data, cdf, marker='.', linestyle='solid')
    plt.xlabel("Data Values")
    plt.ylabel("CDF")
    plt.title("Cumulative Distribution Function (CDF)")
    plt.grid(True)
    plt.show()

    # # 计算KDE估算
    # kde = gaussian_kde(data)
    # x = np.linspace(min(data), max(data), 1000)
    # cdf = [kde.integrate_box_1d(min(data), val) for val in x]

    # # 绘制平滑的CDF曲线
    # plt.plot(x, cdf)
    # plt.xlabel("Data Values")
    # plt.ylabel("CDF")
    # plt.title("Smoothed Cumulative Distribution Function (CDF)")
    # plt.grid(True)

    # 设定横坐标范围
    # plt.xlim([0.1, 1])  # 设定横坐标范围为[-3, 3]，根据需要进行调整

    # plt.show()

init_beacons()
test()
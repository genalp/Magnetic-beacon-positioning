import numpy as np
import math
import random
import matplotlib.pyplot as plt

class magnetic_beacon:
    def __init__(self 
                 ,M = 300000.0                  # 3000 Am^2(10A 300n 1m*1m)  nT
                 ,sensor_position = [8, 8, 20]  # 传感器坐标
                 ,sensor_fs = 1000              # 传感器采样频率
                 ,if_add_noise = True           # 是否添加噪声
                 ,noise_stddev = 0.1            # 噪声标准差
                 ,noise_max_absolute_value = 18 # 噪声最大值
                 ,signal_len = 2000             # 采样长度
                 ,if_add_angle_error = True     # 是否添加角度误差
                 ,max_angle_error = 1           # 最大角度误差(角度制)
                 ,if_add_beacon_error = True    # 是否添加信标偏心误差
                 ,beacon_error_xy = 0.01        # 信标水平偏心误差
                 ,beacon_error_z = 0.02         # 信标垂直偏心误差
                 ):
        self.M = M                              # 磁信标常数
        self.beacon_number = 0                  # 磁信标数量
        self.beacon_position = []               # 磁信标位置(理想值)
        self.beacon_position_noise = []         # 磁信标位置(噪声值)
        self.beacon_frequency = []              # 磁信标发射频率
        self.per_beacon_signal_ideal = []       # 各个磁信标接收信号(理想值)
        self.sum_signal_ideal = []              # 合信号(理想值)
        self.sum_signal_noise = []              # 合信号(噪声值)
        self.signal_output = []                 # 信号输出

        self.sensor_position = sensor_position  # 传感器位置
        self.sensor_fs = sensor_fs
        self.if_add_noise = if_add_noise
        self.noise_stddev = noise_stddev
        self.noise_max_absolute_value = noise_max_absolute_value
        self.signal_len = signal_len

        self.if_add_angle_error = if_add_angle_error
        self.max_angle_error = max_angle_error

        self.if_add_beacon_error = if_add_beacon_error
        self.beacon_error_xy = beacon_error_xy
        self.beacon_error_z = beacon_error_z
    
    # 设定传感器位置
    def set_sensor_position(self, position):
        self.sensor_position = position

    # 获取传感器位置
    def get_sensor_position(self):
        return self.sensor_position

    # 添加信标
    def add_beacon(self, position, frequency):
        self.beacon_position.append(position)
        self.beacon_frequency.append(frequency)
        self.beacon_number += 1

    # 获取信标坐标信息
    def get_beacon_position(self):
        return self.beacon_position

    # 获取信标数
    def get_beacon_number(self):
        return len(self.beacon_position)
    
    def get_point_in_new_coordinate_system(self, x, y, z, alpha, beta, gamma):
        # 定义旋转矩阵
        Rx = np.array([[1, 0, 0],
                    [0, np.cos(alpha), -np.sin(alpha)],
                    [0, np.sin(alpha), np.cos(alpha)]])
        
        Ry = np.array([[np.cos(beta), 0, np.sin(beta)],
                    [0, 1, 0],
                    [-np.sin(beta), 0, np.cos(beta)]])
        
        Rz = np.array([[np.cos(gamma), -np.sin(gamma), 0],
                    [np.sin(gamma), np.cos(gamma), 0],
                    [0, 0, 1]])
        # 计算总的旋转矩阵
        R = np.dot(Rz, np.dot(Ry, Rx))
        # 将原始点表示为NumPy数组
        original_point = np.array([x, y, z])
        # 计算新点
        new_point = np.dot(original_point, R)
        # 返回新的坐标
        return new_point
    
    # 获取理论磁场振幅
    def get_amplitude(self, K, p0, p1):
        x0, y0, z0 = p0
        x1, y1, z1 = p1
        x = x0 - x1
        y = y0 - y1
        z = z0 - z1

        # 添加角度误差
        if self.if_add_angle_error == True:
            alpha = random.uniform(-1.0*self.max_angle_error, 1.0*self.max_angle_error)
            beta = random.uniform(-1.0*self.max_angle_error, 1.0*self.max_angle_error)
            gamma = random.uniform(-180.0,180.0)
            [x, y, z] = self.get_point_in_new_coordinate_system(x, y, z, np.deg2rad(alpha), np.deg2rad(beta), np.deg2rad(gamma))

        r = np.sqrt(x*x + y*y + z*z)
        return K * np.sqrt(3*z*z + r*r) / (r*r*r*r)
    
    # 获取单个磁信标理论磁场数据
    def get_per_beacon_signal(self, amplitude, frequency):
        # 计算理论数值
        fs = self.sensor_fs  # 采样频率
        n = self.signal_len  # 采样数量
        t = np.linspace(0, n/fs, n, endpoint=False)  # 时间数组
        ideal_signal = amplitude * np.sin(2 * np.pi * frequency * t)
        return ideal_signal

    # 添加噪声
    def add_noise(self, ideal_signal):
        # 添加噪声
        mean = 0  # 噪声的均值
        stddev = self.noise_stddev  # 噪声的标准差
        max_absolute_value = self.noise_max_absolute_value  # 噪声的最大绝对值
        noise = [random.gauss(mean, stddev) for _ in range(len(ideal_signal))]
        noise_scale_factor = max_absolute_value / max([abs(n) for n in noise])  # 对噪声进行缩放
        noise = [n * noise_scale_factor for n in noise]
        noise_signal = [sum(column) for column in zip(noise, ideal_signal)]
        return noise_signal

    # 进行计算
    def run(self):
        if (len(self.beacon_position) != len(self.beacon_frequency)) or (len(self.beacon_position) != self.beacon_number):
            return -1
        if self.beacon_number == 0:
            return -2
        # 初始化计算数据缓存
        self.per_beacon_signal_ideal.clear()
        # self.per_beacon_signal_noise.clear()
        self.sum_signal_ideal.clear()
        self.sum_signal_noise.clear()
        self.signal_output.clear()
        self.beacon_position_noise.clear()
        sensor_position_input = []  # 实际输入的坐标值

        # 判断是否引入信标误差
        if self.if_add_beacon_error == True:
            # 引入信标偏心误差
            for position in self.beacon_position:
                beacon_x = position[0] + random.uniform(-1.0*self.beacon_error_xy, 1.0*self.beacon_error_xy)
                beacon_y = position[1] + random.uniform(-1.0*self.beacon_error_xy, 1.0*self.beacon_error_xy)
                beacon_z = position[2] + random.uniform(-1.0*self.beacon_error_z, 1.0*self.beacon_error_z)
                self.beacon_position_noise.append([beacon_x, beacon_y, beacon_z])
            sensor_position_input = self.beacon_position_noise
        else:
            sensor_position_input = self.beacon_position

        # 逐个计算磁场信号
        for i in range(self.beacon_number):
            amplitude = self.get_amplitude(self.M, self.sensor_position, sensor_position_input[i])
            ideal_signal = self.get_per_beacon_signal(amplitude, self.beacon_frequency[i])
            self.per_beacon_signal_ideal.append(ideal_signal)

        # 求理论和信号
        self.sum_signal_ideal = [sum(column) for column in zip(*self.per_beacon_signal_ideal)]

        # 添加噪声
        if(self.if_add_noise == True):
            self.sum_signal_noise = self.add_noise(self.sum_signal_ideal)
            self.signal_output = self.sum_signal_noise
        else:
            self.signal_output = self.sum_signal_ideal
    
    # 画图
    def draw_signal(self):
        t = np.linspace(0, self.signal_len/self.sensor_fs, self.signal_len, endpoint=False)  # 时间数组

        plt.figure(figsize=(10, 8))

        plt.subplot(2, 1, 1)
        plt.plot(t, self.sum_signal_ideal, label='Original Signal')
        plt.title('Original Signal')

        plt.subplot(2, 1, 2)
        plt.plot(t, self.signal_output, label='Output Signal', color='red')
        plt.title('Output Signal')

        plt.show()

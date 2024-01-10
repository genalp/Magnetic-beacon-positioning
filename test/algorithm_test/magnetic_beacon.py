import numpy as np
import math
import random
import matplotlib.pyplot as plt

class magnetic_beacon:
    def __init__(self
                 ,type = 0                      # 0: 多信标定位  1: 移动信标定位

                #  磁信标系统性质 
                 ,M = 300000.0                  # 3000 Am^2(10A 300n 1m*1m)  nT
                 ,sensor_position = [8, 8, 20]  # 传感器坐标
                 ,sensor_fs = 1000              # 传感器采样频率
                 ,signal_len = 2000             # 采样长度

                #  测量噪声性质
                 ,if_add_noise = True           # 是否添加噪声
                 ,noise_stddev = 0.1            # 噪声标准差
                 ,noise_max_absolute_value = 18 # 噪声最大值

                #  测量系统误差性质
                 ,if_add_angle_error = True     # 是否添加角度误差
                 ,max_angle_error = 1           # 最大角度误差(角度制)
                 ,if_add_beacon_error = True    # 是否添加信标偏心误差
                 ,beacon_error_xy = 0.01        # 信标水平偏心误差
                 ,beacon_error_z = 0.02         # 信标垂直偏心误差

                #  移动信标定位相关参数
                 ,beacon_stop_time = 1          # 信标每到一个点停留时间  s
                 ,beacon_speed = 0.4            # 信标移动速度  m/s
                 ):
        
        self.type = type

        self.M = M                              # 磁信标常数
        self.beacon_number = 0                  # 磁信标数量
        self.beacon_position = []               # 磁信标位置(理想值)
        self.beacon_position_noise = []         # 磁信标位置(噪声值)
        self.beacon_angle_noise = []            # 磁信标角度噪声
        self.beacon_frequency = []              # 磁信标发射频率
        self.per_beacon_signal_ideal = []       # 各个磁信标接收信号(理想值)
        self.sum_signal_ideal = []              # 合信号(理想值)
        self.sum_signal_noise = []              # 合信号(噪声值)
        self.signal_output = []                 # 信号输出

        self.sensor_position = sensor_position  # 传感器位置
        self.sensor_fs = sensor_fs
        self.signal_len = signal_len

        # 添加测量噪声相关
        self.if_add_noise = if_add_noise
        self.noise_stddev = noise_stddev
        self.noise_max_absolute_value = noise_max_absolute_value

        # 添加位置和角度噪声相关
        self.if_add_angle_error = if_add_angle_error
        self.max_angle_error = max_angle_error
        self.if_add_beacon_error = if_add_beacon_error
        self.beacon_error_xy = beacon_error_xy
        self.beacon_error_z = beacon_error_z

        # 移动信标相关
        self.beacon_stop_time = beacon_stop_time
        self.beacon_speed = beacon_speed
        self.time_per_point = []                # 每个点之间需要的时间
        self.time_total = 0                     # 循环一次需要的总时间
        self.timestamp = []                     # 时间戳
        self.mobile_beacon_position = []        # 移动状态下信标位置
        self.amplitude_variation = []           # 动态幅值变化
    
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
        # 判断是否引入信标误差
        if self.if_add_beacon_error == True:
            # 引入信标偏心误差
            beacon_x = position[0] + random.uniform(-1.0*self.beacon_error_xy, 1.0*self.beacon_error_xy)
            beacon_y = position[1] + random.uniform(-1.0*self.beacon_error_xy, 1.0*self.beacon_error_xy)
            beacon_z = position[2] + random.uniform(-1.0*self.beacon_error_z, 1.0*self.beacon_error_z)
            self.beacon_position_noise.append([beacon_x, beacon_y, beacon_z])
        # 添加角度误差
        alpha = random.uniform(-1.0*self.max_angle_error, 1.0*self.max_angle_error)
        beta = random.uniform(-1.0*self.max_angle_error, 1.0*self.max_angle_error)
        gamma = random.uniform(-180.0,180.0)
        self.beacon_angle_noise.append([alpha, beta, gamma])


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
    def get_amplitude(self, K, p0, p1, angle_noise):
        x0, y0, z0 = p0
        x1, y1, z1 = p1
        x = x0 - x1
        y = y0 - y1
        z = z0 - z1

        # 添加角度误差
        if self.if_add_angle_error == True:
            # alpha = random.uniform(-1.0*self.max_angle_error, 1.0*self.max_angle_error)
            # beta = random.uniform(-1.0*self.max_angle_error, 1.0*self.max_angle_error)
            # gamma = random.uniform(-180.0,180.0)
            # print(alpha, beta, gamma)
            [x, y, z] = self.get_point_in_new_coordinate_system(x, y, z, np.deg2rad(angle_noise[0]), np.deg2rad(angle_noise[1]), np.deg2rad(angle_noise[2]))

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
    





    # 以下为移动信标相关
    # 数据预处理
    def mobile_points_data_preprocess(self, points):
        time = []
        sum_time = 0
        points_num = len(points)

        # 计算每个点之间的距离和行走需要的时间
        for i in range(len(points)-1):
            direction = points[i+1] - points[i]
            dis = np.linalg.norm(direction)
            time.append(dis / self.beacon_speed + self.beacon_stop_time)
            sum_time += (dis / self.beacon_speed + self.beacon_stop_time)

        # 计算最后一个点和第一个点之间的距离
        direction = points[points_num-1] - points[0]
        dis = np.linalg.norm(direction)
        time.append(dis / self.beacon_speed + self.beacon_stop_time)
        sum_time += (dis / self.beacon_speed + self.beacon_stop_time)

        return time, sum_time
    

    # 输入时间，判断在该时间点的位置
    def get_point_position(self, points, time):
        time_this_circle = time % self.time_total
        point_index = 0
        # 寻找该点在那段轨迹中
        for per_time in self.time_per_point:
            if (time_this_circle - per_time) <= 0:
                break
            time_this_circle -= per_time
            point_index += 1
        # 对停止时间进行预处理
        if time_this_circle > self.beacon_stop_time:
            time_this_circle -= self.beacon_stop_time
        else:
            time_this_circle = 0
        # 处理返回原点的路径
        if point_index >= (len(points) - 1):
            direction = points[0] - points[len(points) - 1]
            dis = np.linalg.norm(direction)
            # 单位方向向量
            direction = direction / dis
            pos = points[len(points) - 1] + direction * self.beacon_speed * time_this_circle
            return pos
        else:
            direction = points[point_index + 1] - points[point_index]
            dis = np.linalg.norm(direction)
            # 单位方向向量
            direction = direction / dis
            pos = points[point_index] + direction * self.beacon_speed * time_this_circle
            return pos
        
    # 生成一段轨迹
    def create_track(self, points):
        t = np.linspace(0, self.signal_len/self.sensor_fs, self.signal_len, endpoint=False)  # 时间数组
        pos = []
        for time in t:
            pos.append(self.get_point_position(points, time))
        return t, pos

    # 计算幅值动态变化
    def calculate_amplitude_variation(self):
        amplitude = []
        for pos in self.mobile_beacon_position:
            amplitude.append(self.get_amplitude(self.M, self.sensor_position, pos, self.beacon_angle_noise[0]))
        return amplitude




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
        # self.beacon_position_noise.clear()
        beacon_position_input = []  # 实际输入的坐标值

        # 判断是否引入信标误差
        if self.if_add_beacon_error == True:
            # # 引入信标偏心误差
            # for position in self.beacon_position:
            #     beacon_x = position[0] + random.uniform(-1.0*self.beacon_error_xy, 1.0*self.beacon_error_xy)
            #     beacon_y = position[1] + random.uniform(-1.0*self.beacon_error_xy, 1.0*self.beacon_error_xy)
            #     beacon_z = position[2] + random.uniform(-1.0*self.beacon_error_z, 1.0*self.beacon_error_z)
            #     self.beacon_position_noise.append([beacon_x, beacon_y, beacon_z])
            beacon_position_input = self.beacon_position_noise
        else:
            beacon_position_input = self.beacon_position


        # 以下为多信标仿真部分
        if self.type == 0:
            # 逐个计算磁场信号
            for i in range(self.beacon_number):
                amplitude = self.get_amplitude(self.M, self.sensor_position, beacon_position_input[i], self.beacon_angle_noise[i])
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
        
        # 以下为移动信标仿真部分
        elif self.type == 1:
            self.time_per_point, self.time_total = self.mobile_points_data_preprocess(beacon_position_input)
            self.timestamp, self.mobile_beacon_position = self.create_track(beacon_position_input)
            self.amplitude_variation = self.calculate_amplitude_variation()
            # 默认第一个输入的频率是磁信标发射频率
            carrier_frequency = self.beacon_frequency[0]
            # 调幅原始信号
            y_original = np.sin(2 * np.pi * carrier_frequency * self.timestamp)
            # 相乘调幅形成输出信号
            self.signal_output = y_original *  self.amplitude_variation


    # 更新传感器位置
    def update_sensor(self, sensor_position):
        self.sensor_position = sensor_position

    
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

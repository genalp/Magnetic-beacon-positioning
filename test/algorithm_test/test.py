# 调幅测试
# import numpy as np
# import matplotlib.pyplot as plt

# # 生成 x 数据
# t = np.linspace(0, 2, 200, endpoint=False)  # 生成 0 到 1 之间的 1000 个点，表示1秒内的信号

# # 生成调幅指数
# m = 0.5
# # 调制信号
# modulating_frequency = 1  # 调制信号频率
# y_signal = (1 + m * np.sin(2 * np.pi * modulating_frequency * t))


# # 载波信号
# carrier_frequency = 6  # 载波频率
# y_original = np.sin(2 * np.pi * carrier_frequency * t)


# # 生成调幅曲线
# y_amplitude = y_signal * y_original


# # 绘制信号曲线
# plt.figure(figsize=(10, 5))
# plt.subplot(3, 1, 1)
# plt.plot(t, y_signal, label='Signal Signal', color='blue')
# plt.legend()
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')
# plt.title('Amplitude Modulated Signal')


# # 绘制调幅曲线
# plt.subplot(3, 1, 2)
# plt.plot(t, y_amplitude, label='AM Signal', color='blue')
# plt.legend()
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')
# plt.title('Amplitude Modulated Signal')


# # 绘制载波信号
# plt.subplot(3, 1, 3)
# plt.plot(t, y_original, label='Original Signal', color='red')
# plt.legend()
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')
# plt.title('Original Sinusoidal Signal')

# plt.tight_layout()
# plt.show()


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from magnetic_beacon import magnetic_beacon

m = magnetic_beacon()
# 定义一系列三维坐标点
points = np.array([
    [0, 0, 0],
    [2, 0, 0],
    [2, 2, 0],
    [0, 2, 0]
    # [1, 1, 0]
])
# 每个坐标点对应的磁信号大小
B = []
# 停止时间
stop_time = 1.5
# 传感器位置
sensor = np.array([5, 5, 1])
# 速度
speed = 1 # m/s
# 每个点之间需要的时间
time_per_point = []
# 循环一次需要的总时间
time_total = 0
# 信号发射频率
signal_frequency = 5

# 数据预处理
def points_data_preprocess(points, speed):
    time = []
    sum_time = 0
    points_num = len(points)

    # 计算每个点之间的距离和行走需要的时间
    for i in range(len(points)-1):
        direction = points[i+1] - points[i]
        dis = np.linalg.norm(direction)
        time.append(dis / speed + stop_time)
        sum_time += (dis / speed + stop_time)

    # 计算最后一个点和第一个点之间的距离
    direction = points[points_num-1] - points[0]
    dis = np.linalg.norm(direction)
    time.append(dis / speed + stop_time)
    sum_time += (dis / speed + stop_time)

    return time, sum_time


# 输入时间，判断在该时间点的位置
def get_point_position(time):
    time_this_circle = time % time_total
    point_index = 0
    # 寻找该点在那段轨迹中
    for per_time in time_per_point:
        if (time_this_circle - per_time) <= 0:
            break
        time_this_circle -= per_time
        point_index += 1
    
    # 对停止时间进行预处理
    if time_this_circle > stop_time:
        time_this_circle -= stop_time
    else:
        time_this_circle = 0
    # 处理返回原点的路径
    if point_index >= (len(points) - 1):
        direction = points[0] - points[len(points) - 1]
        dis = np.linalg.norm(direction)
        # 单位方向向量
        direction = direction / dis
        pos = points[len(points) - 1] + direction * speed * time_this_circle
        return pos
    else:
        direction = points[point_index + 1] - points[point_index]
        dis = np.linalg.norm(direction)
        # 单位方向向量
        direction = direction / dis
        pos = points[point_index] + direction * speed * time_this_circle
        return pos


# 生成一段轨迹
def create_track(dt, num):
    t = np.arange(0, dt*num, dt)
    pos = []
    for time in t:
        pos.append(get_point_position(time))
    return t, pos


# 计算传感器和信标之间的距离
def cal_sensor_beacon_dis(beacon_pos, sensor_pos):
    dis = []
    for pos in beacon_pos:
        dis.append(np.linalg.norm(pos - sensor_pos))
    return dis


# 调幅
def sign_AM(signal, original):
    # 生成调幅曲线
    amplitude = signal * original
    return amplitude


# 初始化信标
def init_beacons():
    global m
    m = magnetic_beacon(
        type = 1,
        sensor_position = sensor, 
        if_add_noise=False,
        noise_max_absolute_value=10,
        if_add_angle_error=False,
        max_angle_error=1,
        if_add_beacon_error=False,
        beacon_error_xy=0.01,
        beacon_error_z=0.02,
        M=300000,
        beacon_stop_time = stop_time,
        beacon_speed = speed,
        sensor_fs = 100,
        signal_len = 1500
        )
    for beacon in points:
        m.add_beacon(beacon, signal_frequency)
    m.run()
        

# 分隔输出波形
def split_array(input_list, n, m):
    """
    Splits the input_list into a 2D array where each inner array contains m elements,
    taken every n elements from the input_list.

    :param input_list: List of elements to split.
    :param n: Interval to skip elements.
    :param m: Number of elements to take.
    :return: A 2D list where each inner list contains m elements.
    """
    result = []
    for i in range(0, len(input_list), n):
        # Ensure not to go out of bounds and that there are enough elements to form a complete subarray
        if i + m <= len(input_list):
            result.append(input_list[i:i + m])
    return result


# 计算速度并输出停止时间段
def identify_stop_periods_with_absolute_speed(data, speed_threshold):
    """
    Identify stop periods in a list of mileage data considering the absolute value of speed.

    Parameters:
    data (list): A list of mileage data points.
    speed_threshold (float): A threshold for determining when the vehicle is stopped.

    Returns:
    list of tuples: A list of tuples where each tuple represents a stop period in the format (start, end).
    """
    # Calculate the absolute value of speed at each time point
    speeds = [abs(data[i] - data[i - 1]) for i in range(1, len(data))]

    # Identify the stop periods
    stop_periods = []
    start = None

    for i, speed in enumerate(speeds):
        if speed < speed_threshold:
            if start is None:
                start = i  # Mark the start of a stop period
        else:
            if start is not None:
                stop_periods.append((start, i))
                start = None  # Reset the start when the vehicle starts moving again

    # Include the last segment if it ends with a stop
    if start is not None:
        stop_periods.append((start, len(speeds)))

    return stop_periods


# 计算每段平均幅值
def calculate_average_amplitude(amplitudes, stop_periods):
    average_amplitudes = []
    for start, end in stop_periods:
        average_amplitude = sum(amplitudes[start:end]) / (end - start) # 这里暂时舍弃了最后一个数
        average_amplitudes.append(average_amplitude)
    return average_amplitudes



# -------------- 以下是多信标定位部分 --------------------

from sko.SA import SA
beacon_list = []

# 定位模型
def obj_fun(p):
    global m, points, B, beacon_list

    K = m.M
    x, y, z = p

    if x < 0 or y < 0 or z < 0:
        return 1e10

    fn = []
    f = 0

    # 优化模型
    for i in beacon_list:
        fi = np.log(B[i]) - np.log(K) - \
            0.5 * np.log(3*(points[i][2]-z)**2 + (points[i][0]-x)**2 + (points[i][1]-y)**2 + (points[i][2]-z)**2) + \
            2 * np.log((points[i][0]-x)**2 + (points[i][1]-y)**2 + (points[i][2]-z)**2)
        
        # fi = K * np.sqrt(3*(beacons[i][0][2]-z)**2 + (beacons[i][0][0]-x)**2 + (beacons[i][0][1]-y)**2 + (beacons[i][0][2]-z)**2) - \
        #     B[i] * ((beacons[i][0][0]-x)**2 + (beacons[i][0][1]-y)**2 + (beacons[i][0][2]-z)**2)**2

        fn.append(fi)
    
    for fi in fn:
        fi = fi * 1e2
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


# 计算权重
def calculate_weights(n = 1):
    global B, beacon_list
    b = 0
    for i in beacon_list:
        b += np.power(B[i], n)
    # b = np.power(b, n)
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


# 利用多信标算法进行定位
def get_position():
    global beacon_list
    every_position = []
    weights = []
    for i in range(len(points)):
        beacon_list.clear()
        beacon_list = list(range(len(points)))
        beacon_list.pop(i)
        position = SA_position()
        every_position.append(position)
        weights.append(calculate_weights(6))
    # print(np.array(every_position),'\n')
    # num = find_most_deviant_point_index(every_position)
    # every_position.pop(num)
    # weights.pop(num)
    # print(np.array(every_position))
    position = weighted_centroid(every_position, weights)
    return position





# Assuming the data is read from a file, the following function can be used to process it
def process_data(file_path):
    # Read the data from the file
    with open(file_path, 'r') as file:
        data = np.array([list(map(float, row.split(','))) for row in file])

    # Separate the data into three arrays
    x, y, z = data.T

    # Calculate the Euclidean norm (2-norm) of the vectors
    norms = np.linalg.norm(data, axis=1)

    return x, y, z, norms




if __name__ == "__main__":
    # 数据预处理
    # time_per_point, time_total = points_data_preprocess(points, speed)
    # t, pos = create_track(0.01,900)

    init_beacons()
    t = m.timestamp
    pos = m.mobile_beacon_position
    dis = m.amplitude_variation
    out = m.signal_output

    Filterdata_file_path = 'C:\code\code\Magnetic-beacon-positioning\project\Filterdata.txt'
    Fx, Fy, Fz, Fnorms = process_data(Filterdata_file_path)
    out = Fnorms
    

    # 分割信号
    signals = split_array(out, 25, 100)
    # 每个窗口的FFT数值
    amplitudes = []
    # 对每个小信号进行FFT
    for signal in signals:
        fft_result = np.abs(np.fft.fft(signal))
        index = int(signal_frequency * len(fft_result) / m.sensor_fs)
        # 根据FFT的幅值公式将幅值除以FFT数据点数量的一半
        amplitude_at_target_frequency = np.abs(fft_result[index]) / (len(fft_result) / 2)
        amplitudes.append(amplitude_at_target_frequency)

    # print(amplitudes)
    # 寻找停止时段
    stop_times = identify_stop_periods_with_absolute_speed(amplitudes, 3)
    print(stop_times)
    average_amplitudes = calculate_average_amplitude(amplitudes, stop_times)
    # B = average_amplitudes
    print(average_amplitudes)
    # position = get_position()
    # print("position = ", position)
    # n = np.mean(sensor - position)
    # print("error = ",n)


    # fft_result = np.abs(np.fft.fft(out))
    # index = int(signal_frequency * len(fft_result) / m.sensor_fs)
    # amplitude_at_target_frequency = np.abs(fft_result[index]) / (len(fft_result) / 2)
    # print(amplitude_at_target_frequency)
    # print(dis[0])
    

    plt.rcParams['font.sans-serif'] = ['SimHei']

    # 绘制信号曲线
    plt.figure(figsize=(10, 5))
    # plt.subplot(2, 1, 1)
    # plt.plot(t, dis, label='理论幅值', color='blue')
    # plt.legend()
    # plt.xlabel('Time (s)')
    # plt.ylabel('Amplitude')
    # plt.title('Amplitude Modulated Signal')


    # 绘制调幅曲线
    # plt.subplot(2, 1, 2)
    plt.plot(amplitudes, label='解算幅值', color='blue')
    plt.legend()
    # plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Amplitude Modulated Signal')

    plt.tight_layout()
    plt.show()


    # # 分离x、y、z坐标
    # x, y, z = zip(*pos)
    # # 创建一个3D图形
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # # 绘制散点图
    # sensor_scatter = ax.scatter(x, y, z, c='b', marker='o')
    # # Set axis labels
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')
    # # Show the plot
    # plt.show()



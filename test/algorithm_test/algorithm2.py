from magnetic_beacon import magnetic_beacon
from scipy import signal
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
import numpy as np
from sko.PSO import PSO
from sko.SA import SA
from sko.GA import GA
import pandas as pd
from scipy.stats import gaussian_kde
import random
from tqdm import tqdm

beacons_num = 5    # 信标数量
beacons_zone = [    # 信标区域
    [7.5, 12.5]         # X轴坐标范围
    ,[7.5, 12.5]        # Y轴坐标范围
    ,[0, 3]         # Z轴坐标范围
]
sensor_num = 100    # 定位点数量
sensor_zone = [
    [0, 35]
    ,[0, 35]
    ,[1, 20]
]

# sensor_num = 300 
# sensor_zone = [
#     [0, 35]
#     ,[0, 35]
#     ,[1, 10]
# ]

# 初始化信标和定位点
beacons = []
for i in range(beacons_num):
    x = random.uniform(beacons_zone[0][0], beacons_zone[0][1])
    y = random.uniform(beacons_zone[1][0], beacons_zone[1][1])
    z = random.uniform(beacons_zone[2][0], beacons_zone[2][1])
    beacons.append([[x, y, z], (i+1)*2])

sensors = []
for i in range(sensor_num):
    x = random.uniform(sensor_zone[0][0], sensor_zone[0][1])
    y = random.uniform(sensor_zone[1][0], sensor_zone[1][1])
    z = random.uniform(sensor_zone[2][0], sensor_zone[2][1])
    sensors.append([x, y, z])

# beacons = [
#     [[7.5, 7.5, 0], 3]
#     ,[[7.5, 12.5, 0], 4]
#     ,[[12.5, 7.5, 0], 5]
#     ,[[12.5, 12.5, 0], 6]
# ]

# # 分离x、y、z坐标
# x, y, z = zip(*sensors)
# # 创建一个3D图形
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# # 绘制散点图
# sensor_scatter = ax.scatter(x, y, z, c='b', marker='o')

# # 分离x、y、z坐标
# x, y, z, k = zip(*((coord[0][0], coord[0][1], coord[0][2], coord[1]) for coord in beacons))
# # 绘制散点图
# beacon_scatter = ax.scatter(x, y, z, c='r', marker='x')

# # 设置坐标轴标签
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')

# # 添加图例
# # 创建一个空白的 Line2D 对象作为图例的代理
# proxy_sensor = Line2D([0], [0], linestyle='none', c='b', marker='o', label='sensors')
# proxy_beacon = Line2D([0], [0], linestyle='none', c='r', marker='x', label='beacons')
# # 添加图例
# legend1 = ax.legend(handles=[proxy_sensor, proxy_beacon], loc='upper right')
# # 创建一个代理处理程序，将两个图例处理成一个
# ax.add_artist(legend1)

# # 显示图形
# plt.show()

# exit()

B = []
sensor_position = [8, 8, 20]
beacon_list = []
m = magnetic_beacon()
data = []
use_beacon_num = 4      # 使用信标数量

# 1: 经典算法
# 2: 简单质心
# 3: 加权质心(磁感应强度相关)
# 4: 加权质心2
algorithm_num = 1
save_path = 'data1.txt'

# 初始化系统
def init_sys():
    global beacons, sensors, beacons_num, sensor_num
    beacons.clear()
    sensors.clear()
    for i in range(beacons_num):
        x = random.uniform(beacons_zone[0][0], beacons_zone[0][1])
        y = random.uniform(beacons_zone[1][0], beacons_zone[1][1])
        z = random.uniform(beacons_zone[2][0], beacons_zone[2][1])
        beacons.append([[x, y, 0], (i+1)*2])
    for i in range(sensor_num):
        x = random.uniform(sensor_zone[0][0], sensor_zone[0][1])
        y = random.uniform(sensor_zone[1][0], sensor_zone[1][1])
        z = random.uniform(sensor_zone[2][0], sensor_zone[2][1])
        sensors.append([x, y, z])
    beacons = [
    [[7.5, 7.5, 0], 3]
    ,[[7.5, 12.5, 0], 4]
    ,[[12.5, 7.5, 0], 5]
    ,[[12.5, 12.5, 0], 6]
    ,[[10, 10, 0], 8]
]

# 初始化信标
def init_beacons():
    global m, beacons
    m = magnetic_beacon(
        sensor_position = sensor_position, 
        if_add_noise=True,
        noise_max_absolute_value=10,
        if_add_angle_error=True,
        max_angle_error=1,
        if_add_beacon_error=True,
        beacon_error_xy=0.01,
        beacon_error_z=0.02,
        M=300000
        )
    for beacon in beacons:
        m.add_beacon(beacon[0], beacon[1])
    m.run()
    # m.draw_signal()

def draw_sys():
    # 分离x、y、z坐标
    x, y, z = zip(*sensors)
    # 创建一个3D图形
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # 绘制散点图
    ax.scatter(x, y, z, c='b', marker='o')

    # 分离x、y、z坐标
    x, y, z, k = zip(*((coord[0][0], coord[0][1], coord[0][2], coord[1]) for coord in beacons))
    # 绘制散点图
    ax.scatter(x, y, z, c='r', marker='x')

    # 设置坐标轴标签
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # 添加图例
    # 创建一个空白的 Line2D 对象作为图例的代理
    proxy_sensor = Line2D([0], [0], linestyle='none', c='b', marker='o', label='sensors')
    proxy_beacon = Line2D([0], [0], linestyle='none', c='r', marker='x', label='beacons')
    # 添加图例
    legend1 = ax.legend(handles=[proxy_sensor, proxy_beacon], loc='upper right')
    # 创建一个代理处理程序，将两个图例处理成一个
    ax.add_artist(legend1)
    # 显示图形
    plt.show()

# IIR滤波
def beacon_IIR():
    global m
    # 定义IIR低通滤波器的参数
    cutoff_frequency = 60  # 截止频率 (Hz)
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
    # print(B)


# 定位模型
def obj_fun(p):
    global m, beacons, B, beacon_list

    K = m.M
    x, y, z = p

    if x < 0 or y < 0 or z < 0:
        return 1e10

    fn = []
    f = 0

    # 优化模型
    for i in beacon_list:
        fi = np.log(B[i]) - np.log(K) - \
            0.5 * np.log(3*(beacons[i][0][2]-z)**2 + (beacons[i][0][0]-x)**2 + (beacons[i][0][1]-y)**2 + (beacons[i][0][2]-z)**2) + \
            2 * np.log((beacons[i][0][0]-x)**2 + (beacons[i][0][1]-y)**2 + (beacons[i][0][2]-z)**2)
        
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

def GA_position():
    ga = GA(func=obj_fun, n_dim=3, max_iter=1000,
            lb=[0, 0, 0], ub=[30, 30, 15])
    best_params, residuals = ga.run()
    return best_params

# init_beacons()
# filtered_signal = beacon_IIR()
# beacon_FFT(filtered_signal)

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

# 加权质心(论文算法)
def weighted_centroid_paper(positions):
    x = 0
    y = 0
    z = 0
    position = np.mean(np.abs(positions), axis=0)
    omega = []
    for p in positions:
        dis = np.linalg.norm(np.array(p) - np.array(position))
        dis = dis**3
        dis = 1 / dis
        omega.append(dis)
    sum_omega = sum(omega)
    for i in range(len(positions)):
        x += (sum_omega - omega[i]) * positions[i][0]
        y += (sum_omega - omega[i]) * positions[i][1]
        z += (sum_omega - omega[i]) * positions[i][2]
    x /= (len(positions) - 1) * sum_omega
    y /= (len(positions) - 1) * sum_omega
    z /= (len(positions) - 1) * sum_omega
    return [x, y, z]


# 寻找前N大的索引
def find_top_n_indices(input_list, n):
    # 创建一个包含 (索引, 值) 元组的列表
    indexed_list = [(i, value) for i, value in enumerate(input_list)]   
    # 对 indexed_list 根据值进行降序排序
    sorted_list = sorted(indexed_list, key=lambda x: x[1], reverse=True)   
    # 提取前n大的值的索引
    top_n_indices = [index for index, _ in sorted_list[:n]] 
    return top_n_indices

# 寻找偏离最大的点
def find_most_deviant_point_index(points):
    # 计算中心坐标
    center = np.mean(points, axis=0)
    # 计算每个点到中心的欧氏距离
    distances = [np.linalg.norm(np.array(point) - center) for point in points]
    # 找到距离最大的点的索引
    max_deviation_index = np.argmax(distances)
    return max_deviation_index


# 生成球面上的点
def generate_points_on_sphere_with_center(n, r, center):
    points = []
    for _ in range(n):
        while True:
            # 生成极坐标中的角度和高度
            theta = np.random.uniform(0, 2 * np.pi)  # 角度
            phi = np.random.uniform(0, np.pi / 2)  # 高度
            # 将极坐标转换为直角坐标
            x = center[0] + r * np.sin(phi) * np.cos(theta)
            y = center[1] + r * np.sin(phi) * np.sin(theta)
            z = center[2] + r * np.cos(phi)
            if x > 0 and y > 0 and z > 0:
                break
        # 将坐标加入列表
        points.append((x, y, z))
    return points




def test():
    global sensor_position, m, algorithm_num, beacon_list, save_path, sensors, B, use_beacon_num
    data = []
    position = []

    dis = []
    err = []

    for x in tqdm(range(len(sensors)), desc="Processing", unit="sensor"):
    # for sensor in sensors:
        sensor = sensors[x]
        # 数值初始化
        n = 0
        B.clear()
        # 更改传感器位置，重新生成信号
        sensor_position = sensor
        m.set_sensor_position(sensor_position)
        m.run()
        # 滤波和FFT
        filtered_signal = beacon_IIR()
        beacon_FFT(filtered_signal)

        # 检查算法类型
        if algorithm_num == 1:
            beacon_list.clear()
            beacon_list = find_top_n_indices(B, use_beacon_num)
            position = SA_position()
            # position = PSO_position()
            # position = GA_position()
        elif algorithm_num == 2:
            every_position = []
            for i in range(use_beacon_num+1):
                beacon_list.clear()
                beacon_list = find_top_n_indices(B, use_beacon_num+1)
                beacon_list.pop(i)
                position = SA_position()
                every_position.append(position)
            # print(np.array(every_position))
            # num = find_most_deviant_point_index(every_position)
            # every_position.pop(num)
            position = np.mean(np.abs(every_position), axis=0)
        elif algorithm_num == 3:
            every_position = []
            weights = []
            for i in range(use_beacon_num+1):
                beacon_list.clear()
                beacon_list = find_top_n_indices(B, use_beacon_num+1)
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
        elif algorithm_num == 4:
            every_position = []
            for i in range(use_beacon_num+1):
                beacon_list.clear()
                beacon_list = find_top_n_indices(B, use_beacon_num+1)
                beacon_list.pop(i)
                position = SA_position()
                every_position.append(position)
            # print(np.array(every_position))
            position = weighted_centroid_paper(every_position)
        elif algorithm_num == 5:
            every_position = []
            weights = []
            for i in range(use_beacon_num+1):
                beacon_list.clear()
                beacon_list = find_top_n_indices(B, use_beacon_num+1)
                beacon_list.pop(i)
                position = PSO_position()
                every_position.append(position)
                weights.append(calculate_weights(6))
            # print(np.array(every_position),'\n')
            # num = find_most_deviant_point_index(every_position)
            # every_position.pop(num)
            # weights.pop(num)
            # print(np.array(every_position))
            position = weighted_centroid(every_position, weights)
        elif algorithm_num == 6:
            every_position = []
            weights = []
            for i in range(use_beacon_num+1):
                beacon_list.clear()
                beacon_list = find_top_n_indices(B, use_beacon_num+1)
                beacon_list.pop(i)
                position = GA_position()
                every_position.append(position)
                weights.append(calculate_weights(6))
            # print(np.array(every_position),'\n')
            # num = find_most_deviant_point_index(every_position)
            # every_position.pop(num)
            # weights.pop(num)
            # print(np.array(every_position))
            position = weighted_centroid(every_position, weights)

        # print('postion: ',position,'\nsensor:  ', sensor_position)
        for i in range(3):
            n += (np.abs(position[i]) - sensor_position[i])**2
        n = np.sqrt(n)
        # print(n)
        data.append(n)

        # dis.append(np.linalg.norm(np.array(sensor_position) - np.array([10, 10, 1])))
        # err.append(n)
        # # 使用zip函数将dis和err一一对应
        # combined = list(zip(dis, err))
        # # 对combined进行排序，按照dis的值排序
        # sorted_combined = sorted(combined, key=lambda x: x[0])
        # # 将排序后的值重新分开到dis和err列表中
        # sorted_dis, sorted_err = zip(*sorted_combined)

    # 计算CDF
    sorted_data = np.sort(data)
    # cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

    np.savetxt(save_path,sorted_data)

    # 绘制CDF函数图
    # plt.plot(sorted_data, cdf, marker='.', linestyle='solid')
    # plt.xlabel("Data Values")
    # plt.ylabel("CDF")
    # plt.title("Cumulative Distribution Function (CDF)")
    # plt.grid(True)
    # plt.show()

    # plt.plot(sorted_dis, sorted_err, marker='.', linestyle='solid')
    # plt.grid(True)
    # plt.show()
    # np.savetxt(f'dis-{algorithm_num}.txt',sorted_dis)
    # np.savetxt(f'err-{algorithm_num}.txt',sorted_err)

    return np.mean(sorted_data)

# init_beacons()
# # test()
# for i in range(1,5):
#     algorithm_num = i
#     save_path = f'data{0+i}.txt'
#     test()

# beacons_num = use_beacon_num + 1
for i in range(7,8):
    init_sys()
    draw_sys()
    init_beacons()
    for j in [3,5,6]:
        algorithm_num = j
        save_path = f'data{i}-{j}.txt'
        test()

# for i in range(17,19):
#     init_sys()
#     beacons.clear()
#     for j in range(beacons_num):
#         x = random.uniform(beacons_zone[0][0], beacons_zone[0][1])
#         y = random.uniform(beacons_zone[1][0], beacons_zone[1][1])
#         z = random.uniform(beacons_zone[2][0], beacons_zone[2][1])
#         beacons.append([[x, y, 0], (j+1)*2])
#     init_beacons()
#     for j in range(1,5):
#         algorithm_num = j
#         save_path = f'data{i}-{j}.txt'
#         test()

# err_data1 = []
# err_data2 = []
# err_data3 = []
# err_data4 = []
# for i in range(30,45):
#     init_beacons()
#     err_data1.clear()
#     err_data2.clear()
#     err_data3.clear()
#     err_data4.clear()
#     for k in range(1, 40):
#         sensors = generate_points_on_sphere_with_center(30, k, [10, 10, 1.5])
#         for j in range(1,5):
#             algorithm_num = j
#             exec(f"err_data{j}.append(test())")
#     np.savetxt(f'err{i}-1.txt',err_data1)
#     np.savetxt(f'err{i}-2.txt',err_data2)
#     np.savetxt(f'err{i}-3.txt',err_data3)
#     np.savetxt(f'err{i}-4.txt',err_data4)
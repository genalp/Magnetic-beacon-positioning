# import numpy as np
# import matplotlib.pyplot as plt

# # 设置中文字体
# # plt.rcParams['font.sans-serif'] = ['SimHei']

# datanum = 14

# # 从文件中加载第一组数据
# sorted_data1 = np.loadtxt(f'data{datanum}-1.txt')
# cdf1 = np.arange(1, len(sorted_data1) + 1) / len(sorted_data1)
# print("algo1: ", np.mean(sorted_data1))

# # 从文件中加载第二组数据
# sorted_data2 = np.loadtxt(f'data{datanum}-2.txt')
# cdf2 = np.arange(1, len(sorted_data2) + 1) / len(sorted_data2)
# print("algo2: ", np.mean(sorted_data2))

# # 从文件中加载第四组数据
# sorted_data4 = np.loadtxt(f'data{datanum}-4.txt')
# cdf4 = np.arange(1, len(sorted_data4) + 1) / len(sorted_data4)
# print("algo3: ", np.mean(sorted_data4))

# # 从文件中加载第三组数据
# sorted_data3 = np.loadtxt(f'data{datanum}-3.txt')
# cdf3 = np.arange(1, len(sorted_data3) + 1) / len(sorted_data3)
# print("algo4: ", np.mean(sorted_data3))

# # 绘制CDF函数图，使用不同的颜色区分两组数据
# plt.plot(sorted_data1, cdf1, linewidth=2, linestyle='dotted', label='Algorithm 1 - Direct Position', color='green')
# plt.plot(sorted_data2, cdf2, linewidth=2, linestyle='dashed', label='Algorithm 2 - CL', color='blue')
# plt.plot(sorted_data4, cdf4, linewidth=2, linestyle='dashdot', label='Algorithm 3 - WCL', color='orange')
# plt.plot(sorted_data3, cdf3, linewidth=2, linestyle='solid', label='Algorithm 4 - Improved WCL', color='red')

# # 设置 x 和 y 坐标轴的刻度
# # plt.xticks(np.arange(0, 12, 1))
# plt.yticks(np.arange(0, 1.1, 0.1))

# plt.xlabel("positioning error/m")
# plt.ylabel("cumulative probability distribution of the error")
# plt.grid(True)
# plt.legend()
# plt.show()


# 不同算法对精度的影响
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
# plt.rcParams['font.sans-serif'] = ['SimHei']

datanum = 18

# 从文件中加载第一组数据
sorted_data1 = np.loadtxt(f'data{datanum}-3.txt')
cdf1 = np.arange(1, len(sorted_data1) + 1) / len(sorted_data1)
print("SA: ", np.mean(sorted_data1))

# 从文件中加载第二组数据
sorted_data2 = np.loadtxt(f'data{datanum}-5.txt')
cdf2 = np.arange(1, len(sorted_data2) + 1) / len(sorted_data2)
print("PSO: ", np.mean(sorted_data2))

# 从文件中加载第三组数据
sorted_data3 = np.loadtxt(f'data{datanum}-6.txt')
cdf3 = np.arange(1, len(sorted_data3) + 1) / len(sorted_data3)
print("GA: ", np.mean(sorted_data3))

# 从文件中加载第四组数据
sorted_data4 = np.loadtxt(f'data{datanum}-7.txt')
cdf4 = np.arange(1, len(sorted_data3) + 1) / len(sorted_data3)
print("GWO: ", np.mean(sorted_data3))

# 绘制CDF函数图，使用不同的颜色区分两组数据
plt.plot(sorted_data1, cdf1, linewidth=2, linestyle='solid', label='SA', color='red')
plt.plot(sorted_data2, cdf2, linewidth=2, linestyle='dashed', label='PSO', color='blue')
plt.plot(sorted_data3, cdf3, linewidth=2, linestyle='dotted', label='GA', color='green')
plt.plot(sorted_data4, cdf4, linewidth=2, linestyle='dotted', label='GWO', color='orange')

# 设置 x 和 y 坐标轴的刻度
# plt.xticks(np.arange(0, 12, 1))
plt.yticks(np.arange(0, 1.1, 0.1))

plt.xlabel("positioning error/m")
plt.ylabel("cumulative probability distribution of the error")
plt.grid(True)
plt.legend()
plt.show()


# # 不同信标数量影响
# import numpy as np
# import matplotlib.pyplot as plt

# # 设置中文字体
# # plt.rcParams['font.sans-serif'] = ['SimHei']

# datanum = 15

# # 从文件中加载第一组数据
# sorted_data1 = np.loadtxt(f'data{datanum}-1.txt')
# cdf1 = np.arange(1, len(sorted_data1) + 1) / len(sorted_data1)
# print("3: ", np.mean(sorted_data1))

# # 从文件中加载第二组数据
# sorted_data2 = np.loadtxt(f'data{datanum}-2.txt')
# cdf2 = np.arange(1, len(sorted_data2) + 1) / len(sorted_data2)
# print("4: ", np.mean(sorted_data2))

# # 从文件中加载第三组数据
# sorted_data3 = np.loadtxt(f'data{datanum}-3.txt')
# cdf3 = np.arange(1, len(sorted_data3) + 1) / len(sorted_data3)
# print("5: ", np.mean(sorted_data3))

# # 从文件中加载第四组数据
# sorted_data4 = np.loadtxt(f'data{datanum}-4.txt')
# cdf4 = np.arange(1, len(sorted_data4) + 1) / len(sorted_data4)
# print("6: ", np.mean(sorted_data4))


# # 绘制CDF函数图，使用不同的颜色区分两组数据
# plt.plot(sorted_data1, cdf1, linewidth=2, linestyle='dotted', label='3 beacons', color='green')
# plt.plot(sorted_data2, cdf2, linewidth=2, linestyle='solid', label='4 beacons', color='red')
# plt.plot(sorted_data3, cdf3, linewidth=2, linestyle='dashed', label='5 beacons', color='blue')
# plt.plot(sorted_data4, cdf4, linewidth=2, linestyle='dashdot', label='6 beacons', color='orange')


# # 设置 x 和 y 坐标轴的刻度
# # plt.xticks(np.arange(0, 12, 1))
# plt.yticks(np.arange(0, 1.1, 0.1))

# plt.xlabel("positioning error/m")
# plt.ylabel("cumulative probability distribution of the error")
# plt.grid(True)
# plt.legend()
# plt.show()



# # 稳定性试验
# import numpy as np
# import matplotlib.pyplot as plt

# # 读取文件中的数据
# file_path = 'err.txt'  # 替换为你的文件路径
# data = np.loadtxt(file_path)

# # 计算最大值、平均值和方差
# max_value = np.max(data)
# mean_value = np.mean(data)
# variance_value = np.std(data)

# # 打印结果
# print(f"最大值: {max_value}")
# print(f"平均值: {mean_value}")
# print(f"标准差: {variance_value}")

# # 绘制曲线图
# plt.plot(data)

# # 设置图形标题和坐标轴标签
# plt.xlabel('number of tests')
# plt.ylabel('positioning error/m')

# plt.yticks(np.arange(0.026, 0.043, 0.002))
# plt.grid(True)

# # 显示图形
# plt.show()


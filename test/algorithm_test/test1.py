import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# 设置中文字体
# plt.rcParams['font.sans-serif'] = ['SimHei']

datanum = 5

# 从文件中加载第一组数据
sorted_data1 = np.loadtxt(f'data{datanum}-1.txt')
cdf1 = np.arange(1, len(sorted_data1) + 1) / len(sorted_data1)
print("algo1: ", np.mean(sorted_data1))

# 从文件中加载第二组数据
sorted_data2 = np.loadtxt(f'data{datanum}-2.txt')
cdf2 = np.arange(1, len(sorted_data2) + 1) / len(sorted_data2)
print("algo2: ", np.mean(sorted_data2))

# 从文件中加载第四组数据
sorted_data4 = np.loadtxt(f'data{datanum}-4.txt')
cdf4 = np.arange(1, len(sorted_data4) + 1) / len(sorted_data4)
print("algo3: ", np.mean(sorted_data4))

# 从文件中加载第三组数据
sorted_data3 = np.loadtxt(f'data{datanum}-3.txt')
cdf3 = np.arange(1, len(sorted_data3) + 1) / len(sorted_data3)
print("algo4: ", np.mean(sorted_data3))

# 绘制CDF函数图，使用不同的颜色区分两组数据
plt.plot(sorted_data1, cdf1, linestyle='dotted', label='Algorithm 1 - Direct Position', color='green')
plt.plot(sorted_data2, cdf2, linestyle='dashed', label='Algorithm 2 - CL', color='blue')
plt.plot(sorted_data4, cdf4, linestyle='dashdot', label='Algorithm 3 - WCL', color='orange')
plt.plot(sorted_data3, cdf3, linestyle='solid', label='Algorithm 4 - Improved WCL', color='red')

# 设置 x 和 y 坐标轴的刻度
plt.xticks(np.arange(0, 12, 1))
plt.yticks(np.arange(0, 1.1, 0.1))

plt.xlabel("positioning error/m")
plt.ylabel("cumulative probability distribution of the error")
plt.grid(True)
plt.legend()
plt.show()





# import numpy as np
# import matplotlib.pyplot as plt

# # 设置中文字体
# plt.rcParams['font.sans-serif'] = ['SimHei']

# datanum = 35

# # 从文件中加载第一组数据
# err1 = np.loadtxt(f'err{datanum}-1.txt')

# # 从文件中加载第二组数据
# err2 = np.loadtxt(f'err{datanum}-2.txt')

# # 从文件中加载第三组数据
# err3 = np.loadtxt(f'err{datanum}-3.txt')

# # 从文件中加载第四组数据
# err4 = np.loadtxt(f'err{datanum}-4.txt')

# # 绘制CDF函数图，使用不同的颜色区分两组数据
# plt.plot(range(1, 40), err1, marker='.', linestyle='solid', label='经典算法', color='green')
# plt.plot(range(1, 40), err2, marker='.', linestyle='solid', label='简单质心', color='blue')
# plt.plot(range(1, 40), err3, marker='.', linestyle='solid', label='加权质心', color='red')
# plt.plot(range(1, 40), err4, marker='.', linestyle='solid', label='随机取样', color='orange')

# # plt.xlabel("Data Values")
# # plt.ylabel("CDF")
# # plt.title("Cumulative Distribution Function (CDF)")
# plt.grid(True)
# plt.legend()
# plt.show()



import numpy as np
import matplotlib.pyplot as plt

# 从文件中加载第二组数据
sorted_data1 = np.loadtxt('sorted_data1.txt')
cdf1 = np.loadtxt('cdf1.txt')

# 从文件中加载第一组数据
sorted_data2 = np.loadtxt('sorted_data9.txt')
cdf2 = np.loadtxt('cdf9.txt')

# 从文件中加载第二组数据
sorted_data3 = np.loadtxt('sorted_data8.txt')
cdf3 = np.loadtxt('cdf8.txt')

# 绘制CDF函数图，使用不同的颜色区分两组数据
plt.plot(sorted_data1, cdf1, marker='.', linestyle='solid', label='Data 1', color='green')
plt.plot(sorted_data2, cdf2, marker='.', linestyle='solid', label='Data 2', color='blue')
plt.plot(sorted_data3, cdf3, marker='.', linestyle='solid', label='Data 3', color='red')

plt.xlabel("Data Values")
plt.ylabel("CDF")
plt.title("Cumulative Distribution Function (CDF)")
plt.grid(True)
plt.legend()
plt.show()

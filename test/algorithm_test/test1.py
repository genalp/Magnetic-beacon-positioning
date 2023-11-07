import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 极坐标参数
r = 5       # 极径
a = np.pi/8
theta = np.linspace(0, a, 100)  # 极角范围，这里 a 表示夹角
phi = np.linspace(0, np.pi/2, 100)  # 高度角范围，通常为0到π/2

# 创建网格
R, Theta, Phi = np.meshgrid(r, theta, phi)

# 将极坐标转换为直角坐标
X = R * np.sin(Phi) * np.cos(Theta)
Y = R * np.sin(Phi) * np.sin(Theta)
Z = R * np.cos(Phi)

# 创建一个三维图形对象
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制弧面
ax.plot_surface(X, Y, Z, color='b', alpha=0.7)

# 设置坐标轴标签
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# 显示图形
plt.show()

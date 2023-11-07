import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sko.GA import GA
from sko.PSO import PSO
from sko.SA import SA
from sko.SA import SAFast
import pandas as pd
import math
import random

K = 300000.0 # 3000 Am^2(10A 300n 1m*1m)  nT
p1 = [0, 2, 0]
p2 = [1, 0, 0]
p3 = [5, 0, 0]
p4 = [3, 0, 0]
p5 = [1, 1, 0]
p0 = [8, 8, 20]

def get_B(K, p0, p1):
    x0, y0, z0 = p0
    x1, y1, z1 = p1
    x = x0 - x1
    y = y0 - y1
    z = z0 - z1
    r = np.sqrt(x*x + y*y + z*z)
    return K * np.sqrt(3*z*z + r*r) / (r*r*r*r)

def get_distance(p1, p2):
    result = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)
    return result


if_get_B = False
B1 = 0
B2 = 0
B3 = 0
B4 = 0
B5 = 0
def obj_fun(p):
    global K, p1, p2, p3, p4, p5, p0

    global if_get_B, B1, B2, B3, B4, B5

    x, y, z = p
    # x, y = p
    # z = -3

    if if_get_B == False:
        B1 = get_B(K, p1, p0) + random.gauss(0, 1)
        B2 = get_B(K, p2, p0) + random.gauss(0, 1)
        B3 = get_B(K, p3, p0) + random.gauss(0, 1)
        B4 = get_B(K, p4, p0) + random.gauss(0, 1)
        B5 = get_B(K, p5, p0) + random.gauss(0, 1)
        print(B1,B2,B3,B4,B5)
        if_get_B = True

    f1 = np.log(B1) - np.log(K) - 0.5 * np.log(3*(p1[2]-z)**2 + (p1[0]-x)**2 + (p1[1]-y)**2 + (p1[2]-z)**2) + 2 * np.log((p1[0]-x)**2 + (p1[1]-y)**2 + (p1[2]-z)**2)
    f2 = np.log(B2) - np.log(K) - 0.5 * np.log(3*(p2[2]-z)**2 + (p2[0]-x)**2 + (p2[1]-y)**2 + (p2[2]-z)**2) + 2 * np.log((p2[0]-x)**2 + (p2[1]-y)**2 + (p2[2]-z)**2)
    f3 = np.log(B3) - np.log(K) - 0.5 * np.log(3*(p3[2]-z)**2 + (p3[0]-x)**2 + (p3[1]-y)**2 + (p3[2]-z)**2) + 2 * np.log((p3[0]-x)**2 + (p3[1]-y)**2 + (p3[2]-z)**2)
    f4 = np.log(B4) - np.log(K) - 0.5 * np.log(3*(p4[2]-z)**2 + (p4[0]-x)**2 + (p4[1]-y)**2 + (p4[2]-z)**2) + 2 * np.log((p4[0]-x)**2 + (p4[1]-y)**2 + (p4[2]-z)**2)
    f5 = np.log(B5) - np.log(K) - 0.5 * np.log(3*(p5[2]-z)**2 + (p5[0]-x)**2 + (p5[1]-y)**2 + (p5[2]-z)**2) + 2 * np.log((p5[0]-x)**2 + (p5[1]-y)**2 + (p5[2]-z)**2)

    # f1 = K * np.sqrt(3*(p1[2]-z)**2 + (p1[0]-x)**2 + (p1[1]-y)**2 + (p1[2]-z)**2) - B1 * ((p1[0]-x)**2 + (p1[1]-y)**2 + (p1[2]-z)**2)**2
    # f2 = K * np.sqrt(3*(p2[2]-z)**2 + (p2[0]-x)**2 + (p2[1]-y)**2 + (p2[2]-z)**2) - B2 * ((p2[0]-x)**2 + (p2[1]-y)**2 + (p2[2]-z)**2)**2
    # f3 = K * np.sqrt(3*(p3[2]-z)**2 + (p3[0]-x)**2 + (p3[1]-y)**2 + (p3[2]-z)**2) - B3 * ((p3[0]-x)**2 + (p3[1]-y)**2 + (p3[2]-z)**2)**2
    # f4 = K * np.sqrt(3*(p4[2]-z)**2 + (p4[0]-x)**2 + (p4[1]-y)**2 + (p4[2]-z)**2) - B4 * ((p4[0]-x)**2 + (p4[1]-y)**2 + (p4[2]-z)**2)**2
    # f5 = K * np.sqrt(3*(p5[2]-z)**2 + (p5[0]-x)**2 + (p5[1]-y)**2 + (p5[2]-z)**2) - B5 * ((p5[0]-x)**2 + (p5[1]-y)**2 + (p5[2]-z)**2)**2

    f1 = f1 * 1e3
    f2 = f2 * 1e3
    f3 = f3 * 1e3
    f4 = f4 * 1e3
    f5 = f5 * 1e3

    f = f1**2+f2**2+f3**2+f4**2
    if f == 0:
        f = 1e-100
    f = np.log10(f)

    return f


# 绘制全局数组的图表
def plot_global_array():
    global every_result
    plt.plot(every_result)
    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.title("Global Dynamic Array")
    plt.show()

def test_model():
    global K, p1, p2, p3, p4, p5, p0

    B1 = get_B(K, p1, p0)
    B2 = get_B(K, p2, p0)
    B3 = get_B(K, p3, p0)
    B4 = get_B(K, p4, p0)
    B5 = get_B(K, p5, p0)

    r1 = math.exp((np.log(B1) - np.log(K))/(-3))
    r2 = math.exp((np.log(B2) - np.log(K))/(-3))
    r3 = math.exp((np.log(B3) - np.log(K))/(-3))
    r4 = math.exp((np.log(B4) - np.log(K))/(-3))
    r5 = math.exp((np.log(B5) - np.log(K))/(-3))

    print(r1, r2, r3, r4, r5)
    print(get_distance(p1,p0), get_distance(p2,p0), get_distance(p3,p0), get_distance(p4,p0), get_distance(p5,p0))


# ga = GA(func=obj_fun, n_dim=3, max_iter=1000,
#         lb=[0, 0, -10], ub=[20, 20, 0])
# best_params, residuals = ga.run()
# print(' x :', best_params[0], ' y :', best_params[1], ' z :', best_params[2], '\n', 'best_y:', residuals)

pso = PSO(func=obj_fun, n_dim=3, 
          pop=50, max_iter=500, 
          lb=[0, 0, 5], ub=[20, 20, 50], 
          w=0.9, c1=0.5, c2=0.5)
pso.run()
print(' x :', pso.gbest_x[0], ' y :', pso.gbest_x[1], ' z: ', pso.gbest_x[2], 'best_y is', pso.gbest_y)
# plt.plot(pso.gbest_y_hist)
# plt.show()

# sa = SA(func=obj_fun, x0=[0, 0, 0], T_max=1, T_min=1e-9, L=300, max_stay_counter=150)
# best_x, best_y = sa.run()
# print('best_x:', best_x, 'best_y', best_y)
# plt.plot(pd.DataFrame(sa.best_y_history).cummin(axis=0))
# plt.show()

# test_model()

# plot_global_array()

# # 创建一定范围内的x和y值
# x = np.linspace(8, 12, 301)
# y = np.linspace(8, 12, 301)
# X, Y = np.meshgrid(x, y)
# Z = obj_fun([X, Y])

# # 创建一个3D图形
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # 绘制立体图像
# ax.plot_surface(X, Y, Z, cmap='viridis')

# # 设置坐标轴标签
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')

# # 显示图形
# plt.show()
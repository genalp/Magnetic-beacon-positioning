import numpy as np
import matplotlib.pyplot as plt
from sko.GA import GA

x_true = np.linspace(-1.2, 1.2, 30)
y_true = x_true ** 3 - x_true + 0.4 * np.random.rand(30)
plt.plot(x_true, y_true, 'o')

def f_fun(x, a, b, c, d):
    return a * x ** 3 + b * x ** 2 + c * x + d


def obj_fun(p):
    a, b, c, d = p
    residuals = np.square(f_fun(x_true, a, b, c, d) - y_true).sum()
    return residuals

ga = GA(func=obj_fun, n_dim=4, size_pop=100, max_iter=500,
        lb=[-2] * 4, ub=[2] * 4)

best_params, residuals = ga.run()
print('best_x:', best_params, '\n', 'best_y:', residuals)

y_predict = f_fun(x_true, *best_params)

fig, ax = plt.subplots()

ax.plot(x_true, y_true, 'o')
ax.plot(x_true, y_predict, '-')

plt.show()
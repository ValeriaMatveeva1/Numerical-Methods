import math
import numpy as np
import matplotlib.pyplot as plt


# функция для выдачи значений функции по списку аргументов
def get_points(x_p, func=None):
    n = x_p.shape[0]
    if func:
        return np.array([func(x_p[i]) for i in range(n)], dtype=float)
    else:
        return np.random.sample(n)


# функция для повышения порядка конечных разностей
def increase_degree(y, i=0):
    for j in range(y.shape[0] - 1 - i):
        y[j] = y[j + 1] - y[j]


# функция интерполирования в точке
def bessel_interpolate_point(x_old, y_old, x_new):
    y = y_old.copy()
    k = x_old.shape[0]
    n = (k - 1) // 2
    h = x_old[1] - x_old[0]
    q = (x_new - x_old[n]) / h
    qc = 1.0
    res = (y[n + 1] + y[n]) * 0.5
    increase_degree(y)
    res += y[n] * (q - 0.5)
    for i in range(n):
        increase_degree(y, i)
        k1 = (y[n - i] + y[n - i - 1]) * 0.5
        increase_degree(y, i + 1)
        k2 = y[n - i - 1]
        qc *= (q + i) * (q - i - 1) / (2 * i + 2)
        res += qc * (k1 + k2 * (q - 0.5) / (2 * i + 3))
        qc /= (2 * i + 3)
    return res


# функция интерполирования в точке или в точках
def bessel_interpolate(x_old, y_old, x_new):
    try:
        res = np.zeros(len(x_new))
        for i in range(res.shape[0]):
            res[i] = bessel_interpolate_point(x_old, y_old, x_new[i])
        return res
    except:
        return bessel_interpolate_point(x_old, y_old, x_new)


# тесты
def test_1():
    x = np.linspace(-5, 5, 12)
    y = get_points(x)  # здесь случайные значения
    x_n = np.linspace(-5, 5, 300)
    y_n = bessel_interpolate(x, y, x_n)
    plt.plot(x_n, y_n)
    plt.plot(x, y)
    plt.show()


def test_2():
    x = np.linspace(-5, 5, 12)
    y = get_points(x, lambda x: 10 * math.sin(x) - x * x)
    x_n = np.linspace(-5, 5, 300)
    y_n = bessel_interpolate(x, y, x_n)
    y_o = get_points(x_n, lambda x: 10 * math.sin(x) - x * x)
    plt.plot(x_n, y_n)
    plt.plot(x_n, y_o)
    plt.show()


def test_3():
    x = np.linspace(-5, 5, 14)
    y = get_points(x, lambda x: 10 * math.exp(-x * x) + x / 10)
    x_n = np.linspace(-5, 5, 300)
    y_n = bessel_interpolate(x, y, x_n)
    y_o = get_points(x_n, lambda x: 10 * math.exp(-x * x) + x / 10)
    plt.plot(x_n, y_n)
    plt.plot(x_n, y_o)
    plt.show()


test_1()
test_2()
test_3()

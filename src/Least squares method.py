from math import *
import numpy as np
import matplotlib.pyplot as plt


# функция для выдачи значений функции по списку аргументов
def get_points(x_p, func=None):
    n = x_p.shape[0]
    if func:
        return np.array([func(x_p[i]) for i in range(n)], dtype=float)
    else:
        return np.random.sample(n)


# линейно независимые функции
def func(i, x):
    return pow(x + 1, i)


# метод, находящий коэффициенты a
def _least_squares_method(x, y, n):
    l = len(x)
    gram = np.matrix([[sum(func(k, x[j]) * func(i, x[j]) for j in range(l)) for k in range(n)] for i in range(n)])
    coef = np.array([sum(y[j] * func(i, x[j]) for j in range(l)) for i in range(n)])
    a = np.linalg.solve(gram, coef)
    return a


# функция получения значений итоговой функции в требуемых точках
def least_squares_method(x, y, x_n, n):
    res = np.zeros(len(x_n))
    a = _least_squares_method(x, y, n)
    for i in range(res.shape[0]):
        res[i] = sum(func(j, x_n[i]) * a[j] for j in range(n))
    return res


def test_1():
    x = np.linspace(-5, 5, 200)
    y = get_points(x, lambda x: sin(20 * x) + sin(x) + x)
    x_n = np.linspace(-5, 5, 100)
    y_n = least_squares_method(x, y, x_n, 5)
    plt.plot(x_n, y_n)
    plt.plot(x, y)
    plt.show()


def test_2():
    x = np.linspace(5, 20, 200)
    y = get_points(x, lambda x: sin(20 * x) + cos(x) + 4 * log(x))
    x_n = np.linspace(5, 20, 100)
    y_n = least_squares_method(x, y, x_n, 5)
    plt.plot(x_n, y_n)
    plt.plot(x, y)
    plt.show()


test_1()
test_2()

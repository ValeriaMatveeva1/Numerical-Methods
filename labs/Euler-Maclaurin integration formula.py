from math import *
from scipy.misc import derivative
from scipy.integrate import quad


def euler_maclaurin(f, a, b, n):
    h = (b - a) / (n - 1)
    res = (f(a) + f(b)) / 2
    for i in range(1, n - 1):
        res += f(a + i * h)
    res *= h
    res += h ** 2 * (derivative(f, a, dx=1e-6, n=1) - derivative(f, b, dx=1e-6, n=1)) / 12
    res -= h ** 4 * (derivative(f, a, dx=1e-6, n=3, order=5) - derivative(f, b, dx=1e-6, n=3, order=5)) / 720
    res += h ** 6 * (derivative(f, a, dx=1e-6, n=5, order=7) - derivative(f, b, dx=1e-6, n=5, order=7)) / 30240
    return res


def test_1():
    a = -4
    b = 8
    n = 10000
    f = (lambda x: e ** (-x ** 2) + x * sin(x))
    r1 = euler_maclaurin(f, a, b, n)
    r2 = quad(f, a, b)[0]
    print("Найденное значение: ", r1, "\nОшибка: ", abs(r1 - r2))


def test_2():
    a = 10
    b = 15
    n = 10000
    f = (lambda x: log(x)/x + x**2)
    r1 = euler_maclaurin(f, a, b, n)
    r2 = quad(f, a, b)[0]
    print("Найденное значение: ", r1, "\nОшибка: ", abs(r1 - r2))


test_1()
test_2()
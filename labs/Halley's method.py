import math
import numpy as np
from scipy.misc import derivative


def halleys_mathod(f, n: int) -> np.array:
    r = 100  # делаем невязку достаточно большой в начале, чтобы определить работу функции
    x = 0  # начальное приближение x0
    if n > 0:
        for i in range(n):
            df = derivative(f, x, dx=1e-6, n=1)
            ddf = derivative(f, x, dx=1e-6, n=2)
            t = - f(x) / df  # Ньютоновская поправка
            r = (ddf * t * t) / df  # невязка
            if abs(r) < 10e-12:  # остановка цикла по достижению нужного результата
                break
            x = x + (t * t) / (t + r / 2)  # вычисляем k-ое приближение по формуле
        print("Error: ", abs(r))
        print("Root: ", end='')
        return x  # корень нелинейного уравнения
    else:
        print("A number of iterations can't be a non-positive value!")


def test_1():
    f = lambda x: x**3 + 6*x**2 + 9*x - 4
    print("Current function: x\u00b3+6*x\u00b2+9*x-4".format(''))
    print(halleys_mathod(f, 10))


def test_2():
    f = lambda x: math.sin(x) + 10 - x**2
    print("Current function: sin(x)+10-x\u00b2".format(''))
    print(halleys_mathod(f, 40))


def test_3():
    f = lambda x: 1 / (10 + x**2) + x
    print("Current function: (1 / 10+x\u00b2)+x".format(''))
    print(halleys_mathod(f, 100))


def test_4():
    f = lambda x: math.log(x + 1) + math.cos(x)
    print("Current function: log(x+1)+cos(x)")
    print(halleys_mathod(f, 200))


def test_5():
    f = lambda x: math.acos(x) - x**3 - 1
    print("Current function: arccos(x)-x\u00b3-1".format(''))
    print(halleys_mathod(f, 1000))


test_1()
test_2()
test_3()
test_4()
test_5()


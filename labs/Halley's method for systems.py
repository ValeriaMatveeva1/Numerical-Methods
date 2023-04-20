import math
import numpy as np
from scipy.misc import derivative

list_of_variables = ['x', 'y', 'z']


def halleys_method(F: list, x: np.array, n: int, eps: float):
    size = len(F)
    vars = list_of_variables[0:size]
    if n > 0:
        for i in range(n):
            f_x = np.empty((0, size), float)  # массив для значений ф-ций
            df_x = np.empty((0, size), float)  # массив для формирования матрицы Якоби
            ddf_x = np.empty((0, size), float)  # матрица для формирования матрицы Гессе
            for j in range(size):  # итерируемся по функциям и находим f(x), f'(x), f''(x), формируем матрицы
                f_x = np.append(f_x, F[j](*x))
                d = [[df(F[j], param, x, 1, size) for param in vars]]
                dd = [[df(F[j], param, x, 2, size) for param in vars]]
                df_x = np.append(df_x, d, axis=0)
                ddf_x = np.append(ddf_x, dd, axis=0)
            if np.linalg.norm(f_x) < eps:  # критерий остановки
                break
            print('Iteration', i,  ':', vars, '=', x)
            t = jordan_method(df_x, -f_x)  # Ньютоновская поправка
            tt = np.array([i * i for i in t])
            ddf_x_tt = ddf_x.dot(tt)
            r = jordan_method(df_x, ddf_x_tt)  # Чебышевская поправка
            x = x + np.array([tt[i] / (t[i] + 0.5 * r[i]) for i in range(tt.shape[0])])  # i-тое приближение вектора x
        print('Error: ', [i(*x) for i in F])  # невязка
        print('Result: ', vars, '= ', end='')
        return x
    else:
        print("A number of iterations can't be a non-positive value!")


def jordan_method(a: np.array, f: np.array) -> np.array:
    size: int = a.shape[0]  # размер матрицы
    A = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            A[i, j] = a[i, j]
    r = f.copy()  # копия (для дальнейшего вывода без проблем)
    if np.linalg.det(A) != 0:
        for i in range(0, size):
            r[i] = r[i] / A[i, i]  # делим каждую строку на число на главной диагонали
            A[i] = A[i] / A[i, i]
            for j in range(i + 1, size):  # из нижележащих строк вычитаем i-тую, умноженную на i-тый эл-т j-той строки
                # так же изменяем вектор
                r[j] = r[j] - r[i] * A[j, i]
                A[j] = A[j] - A[i] * A[j, i]
            for k in range(i - 1, -1, -1):  # из вышележащих строк вычитаем i-тую, умноженную на i-тый эл-т k-той строки
                # так же изменяем вектор правой части
                r[k] = r[k] - r[i] * A[k, i]
                A[k] = A[k] - A[i] * A[k, i]
        return r
    else:
        print("Det(A) = 0!")


def df(f, param: str, p: np.array, n: int, size: int):
    t = 1e-6
    if size == 2:
        if param == 'x':
            return derivative(lambda x: f(x, 0), p[0], dx=t, n=n)
        elif param == 'y':
            return derivative(lambda y: f(0, y), p[1], dx=t, n=n)
    elif size == 3:
        if param == 'x':
            return derivative(lambda x: f(x, 0, 0), p[0], dx=t, n=n)
        elif param == 'y':
            return derivative(lambda y: f(0, y, 0), p[1], dx=t, n=n)
        elif param == 'z':
            return derivative(lambda z: f(0, 0, z), p[2], dx=t, n=n)


def test_1():
    var1 = lambda x, y: x ** 2 - y - 1
    var2 = lambda x, y: x - y ** 2 + 1
    print("Current system:\n{{x\u00b2-y-1=0,\nx-y\u00b2+1=0}}".format(''))
    f = [var1, var2]
    x0 = np.array([1, 2])
    print("x0: ", x0)
    n = 10
    print("n: ", n)
    eps = 1e-6
    print(halleys_method(f, x0, n, eps))


def test_2():
    var1 = lambda x, y: x - y - 8
    var2 = lambda x, y: x ** 2 + y ** 2 - 82
    print("Current system:\n{{x-y-8=0,\nx\u00b2+y\u00b2-82=0}}".format(''))
    f = [var1, var2]
    x0 = np.array([7.5, 2.5])
    print("x0: ", x0)
    n = 60
    print("n: ", n)
    eps = 1e-6
    print(halleys_method(f, x0, n, eps))


def test_3():
    var1 = lambda x, y, z: x ** 2 + y ** 2 + z ** 2 - 1
    var2 = lambda x, y, z: 2 * x ** 2 + y ** 2 - 4 * z ** 2
    var3 = lambda x, y, z: 3 * x ** 2 - 4 * y ** 2 + z ** 2
    print("Current system:\n{{x\u00b2+y\u00b2+z\u00b2-1=0,\n2x\u00b2+y\u00b2-4z\u00b2=0,\n3x\u00b2-4y\u00b2+z\u00b2=0}}".format(''))
    f = [var1, var2, var3]
    x0 = np.array([0.5, 0.5, 0.5])
    print("x0: ", x0)
    n = 100
    print("n: ", n)
    eps = 1e-6
    print(halleys_method(f, x0, n, eps))


def test_4():
    var1 = lambda x, y: math.sin(x) - y - 1.32
    var2 = lambda x, y: math.cos(y) - x + 0.35
    print("Current system:\n{sin(x)-y-1.32=0,\ncos(y)-x+0.35=0}")
    f = [var1, var2]
    x0 = np.array([1.8, -0.3])
    print("x0: ", x0)
    n = 100
    print("n: ", n)
    eps = 1e-6
    print(halleys_method(f, x0, n, eps))


test_1()
test_2()
test_3()
test_4()

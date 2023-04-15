import numpy as np


def modified_inverse_iteration_method(A: np.array, alpha: float, n: int) -> tuple:
    A_copy = A.copy()
    pleasant_output(A, 2)
    print("\u03B1: ", alpha)
    size = A_copy.shape[0]
    I = np.eye(size)
    x = np.ones(size)
    l = 0
    r = 0
    for i in range(n):
        y_kplus1 = np.linalg.solve(A_copy - alpha * I, x)  # находим y(k+1), решая систему
        x = y_kplus1 / np.linalg.norm(y_kplus1)  # производим нормировку вектора
        l = get_eigenvalue(A, x)
        r = np.linalg.norm(np.dot(A, x) - np.dot(get_eigenvalue(A, x), x))  # норма вектора невязки
        if r < 1e-5:  # критерий остановки
            print("Number of the last iteration: ", i + 1)
            break
    print("Norm of nullity vector: ", r)
    print("Result eigenvector and eigenvalue: ")
    return x.tolist(), l


def pleasant_output(M: np.array, dim: int) -> None:
    if dim == 1:
        print("F: ")
        for x in range(M.shape[0]):
            print("{0:7.3f} ".format(M[x]), end="")
    if dim == 2:
        print("A: ")
        for m in range(M.shape[0]):
            for n in range(M.shape[1]):
                print("{0:7.3f} ".format(M[m, n]), end="")
            print()
    print()


def get_eigenvalue(A: np.array, x: np.array):
    r = np.dot(A, x)  # получаем вектор правой части (A*x)
    res = sum(r[i] / x[i] for i in range(x.shape[0])) / x.shape[0]  # делим покомпонентно вектор r на x
    # и находим среднее арифметическое полученного вектора
    return res


def test_1():
    A = np.array([
        [2, -9, 5],
        [1.2, -5.3999, 6],
        [1, -1, -7.5]
    ])
    alpha = -8
    res = modified_inverse_iteration_method(A, alpha, 1000)
    print(res)


def test_2():
    A = np.array([
        [1.00, 0.77, -0.25, 0.54],
        [0.77, 1.00, 0.35, 0.43],
        [-0.25, 0.35, 1.00, -0.74],
        [0.54, 0.43, -0.74, 1.00]
    ])
    alpha = 2.5
    res = modified_inverse_iteration_method(A, alpha, 100)
    print(res)


def test_3():
    A = np.array([
        [1, 0.42, 0.54, 0.66],
        [0.42, 1.00, 0.32, 0.44],
        [0.54, 0.34, 1.00, 0.22],
        [0.66, 0.44, 0.22, 1.00]
    ])
    alpha = -1
    res = modified_inverse_iteration_method(A, alpha, 50)
    print(res)


test_1()
test_2()
test_3()

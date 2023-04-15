import numpy as np


def minimal_residual_method(A: np.array, f_: np.array, n: int) -> np.array:
    # в данной реализации считается, что на входе матрица положительно определена
    f = f_.copy()  # копия правой части, чтобы не портить входные данные
    A_transposed = np.transpose(A.copy())
    B = np.dot(A, A_transposed)  # матрица (A*A_transposed), т.к. при 2 трансформации Гаусса
    # решается уравнение (A*A_transposed) * y = f, а далее находится x = A_transposed * f
    y = np.zeros(f.shape, dtype=np.float64)  # начальное приближение
    if n < 0:
        print("A number of iterations can't be non positive.")
    else:
        for i in range(n):
            r = f - np.dot(B, y)  # вектор невязки
            print("Number of iteration: ", i + 1)
            print("x: ", np.dot(A_transposed, y))  # последовательные приближения
            print("Norm of r: ", np.linalg.norm(r))  # норма вектора невязки
            print("")
            B_r = np.dot(B, r)
            alpha = np.dot(B_r, r) / np.dot(B_r, B_r)  # итерационный параметр
            # (alpha выбирается так, чтобы минимизировать евклидову норму невязки)
            y = y + alpha * r  # n-ое приближенное значение решения
    print("Result (x = A * y): ")
    return np.dot(A_transposed, y)  # столбец решений


def test_1():
    A = np.array([[1.00, 0.17, -0.25, 0.54],
                   [0.47, 1.00, 0.67, -0.32],
                   [-0.11, 0.35, 1.00, -0.74],
                   [0.55, 0.43, 0.36, 1.00]], dtype=np.float64)
    f = np.array([0.3, 0.5, 0.7, 0.9], dtype=np.float64)
    print("A: ")
    for m in range(A.shape[0]):
        for k in range(A.shape[1]):
            print("{0:7.3f} ".format(A[m, k]), end="")
        print()
    print("f: ")
    for x in range(f.shape[0]):
        print("{0:7.3f} ".format(f[x]), end="")
    print(" ")

    x = minimal_residual_method(A, f, 100)
    print(x)


test_1()

import numpy as np
import math


def jordan_method(A: np.array, f: np.array) -> np.array:
    r = f.copy()  # копия (для дальнейшего вывода без проблем)
    size: int = A.shape[0]
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


def test_1():
    A = np.array([[1.0, 0.42, 0.54, 0.66],
                   [0.42, 1.00, 0.32, 0.44],
                   [0.54, 0.32, 1.0, 0.22],
                   [0.66, 0.44, 0.22, 1.0]], dtype=np.dtype(np.float32))
    F = np.array([0.3, 0.5, 0.7, 0.9], dtype=np.dtype(np.float32))
    print("\nИсходные данные: ")
    pleasant_output(A, 2)
    pleasant_output(F, 1)
    print("Ответ: \n", jordan_method(A, F))


def test_2():
    A = np.array([[0.5, 1.75, 3.44],
                   [2.43, 8.1, 9.34],
                   [9.11, 0.54, 3.567]], dtype=np.dtype(np.float32))
    F = np.array([1.73, 0.5, 2.7], dtype=np.dtype(np.float32))
    print("\nИсходные данные: ")
    pleasant_output(A, 2)
    pleasant_output(F, 1)
    print("Ответ: \n", jordan_method(A, F))


def test_3():
    A = np.array([[1.0, 0.17, -0.25, 0.54],
                   [0.47, 1.0, 0.67, -0.32],
                   [-0.11, 0.35, 1.0, -0.74],
                   [0.55, 0.43, 0.36, 1.0]])
    F = np.array([0.3, 0.5, 0.7, 0.9], dtype=np.dtype(np.float32))
    print("\nИсходные данные: ")
    pleasant_output(A, 2)
    pleasant_output(F, 1)
    print("Ответ: \n", jordan_method(A, F))


test_1()
test_2()
test_3()

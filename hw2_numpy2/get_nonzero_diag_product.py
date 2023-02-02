"""
Задача A
get_nonzero_diag_product
Напишите функцию get_nonzero_diag_product(X), которая подсчитывает произведение ненулевых элементов на диагонали
прямоугольной матрицы. Если все элементы на диагонали нулевые, функция должна вернуть None.
Функция должна быть написана с использованием библиотеки numpy. Использование циклов категорически запрещено.
Пример:
Входные данные
get_nonzero_diag_product(np.array([[1, 0], [0, 0]]))
Результат работы
>>>1
"""
import numpy as np


def get_nonzero_diag_product(X):
    non_zero_arr = np.diag(X)[np.diag(X) > 0]
    if len(non_zero_arr) > 0:
        return np.prod(non_zero_arr)
    return None

"""
Задача D
get_max_after_zero
Напишите get_max_after_zero(x), возвращающую максимальный элемент в векторе x среди элементов, перед которыми стоит
нулевой. Если подходящих элементов нет, функция должна возвращать None.
Функция должна быть написана с использованием библиотеки numpy. Использование циклов категорически запрещено.
Пример:
Входные данные
get_max_after_zero(np.array([1, 2, 9, 8, 0, 5]))
Результат работы
>>>5
"""
import numpy as np


def get_max_after_zero(x):
    x_size = x.size
    if x_size and 0 in x:
        condition = np.where(x == 0)[0] + 1
        condition = condition[condition < x_size]
        if condition.size:
            return np.max(x[condition])
    return None

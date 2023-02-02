"""
Задача F
encode_rle_numpy
Напишите функцию encode_rle(x), реализующую кодирование длин серий (Run-length encoding). По входному вектору x
необходимо вернуть кортеж из двух векторов одинаковой длины. Первый содержит числа, а второй — сколько раз их нужно
повторить.
Функция должна быть написана с использованием библиотеки numpy. Использование циклов категорически запрещено.
Пример:
Входные данные
encode_rle(np.array([0, 0, 1, 1, 1, 2, 1]))
Результат работы
>>>(np.array([0, 1, 2, 1]), np.array([2, 3, 1, 1]))
"""
import numpy as np


def encode_rle(x):
    values, inverses = np.unique(x, return_inverse=True)
    x_size = x.size
    mask = np.hstack([np.array([True]), (inverses[:-1] - inverses[1:]) != 0])
    rle_values = values[inverses[mask]]
    where_digit_changes = np.where(mask)[0]
    ret_size = rle_values.size
    rle_counts = np.ones(ret_size, dtype=int)
    rle_counts[:ret_size - 1] = where_digit_changes[1:] - where_digit_changes[:-1]
    rle_counts[-1] = x_size - where_digit_changes[-1]
    return rle_values, rle_counts

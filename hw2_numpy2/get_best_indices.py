"""
Задача H
get_best_indices
Часть 1:
На вход вашей программе подается бинарный файл input.bin, который содержит в себе двумерный numpy-массив. Все элементы в
строке массива уникальны.
Напишите программу, которая найдет индексы топ-5 наибольших значений в каждой строке. Индексы должны быть отсортированы
в порядке уменьшения значений, хранящихся по этим индексам.
Ваша программа должна содержать функцию get_best_indices, где ranks – исходный двумерный массив, top – размер топа.
Часть 2:
Теперь будем полагать, что в файле input.bin хранится N-мерный numpy-массив (N ≥ 2).
Поддержите параметр axis, который указывает, вдоль какой оси должно нужно отбирать индексы. Легко убедиться в том, что в
случае N = 2 и axis = 1 имеем базовую формулировку задачи.
Формат входных данных:
numpy-массив в бинарном формате. Для чтения пользуйтесь функцией np.load.
Формат результата:
numpy-массив в бинарном формате. Для записи пользуйтесь функцией np.save.
Примеры:
1) Входные данные в файле input.bin
array([[ 0,  1,  2,  3,  4],
       [ 5,  6,  7,  8,  9],
       [10, 11, 12, 13, 14]])
Результат работы в файле output.bin
>>>array([[4, 3, 2, 1, 0],
       [4, 3, 2, 1, 0],
       [4, 3, 2, 1, 0]])
2) Входные данные в файле input.bin
array([[16, 10,  3,  8, 14,  5,  6, 18,  4, 17, 12,  2, 19,  1,  0],
       [ 6,  4, 19, 15, 13, 11, 14,  0,  7, 18,  9, 12,  8, 17,  1],
       [ 9, 12,  1,  2, 11, 17, 19,  8, 13, 15, 16, 10,  0, 18,  7]])
Результат работы в файле output.bin
>>>array([[12,  7,  9,  0,  4],
       [ 2,  9, 13,  3,  6],
       [ 6, 13,  5, 10,  9]])
"""
import numpy as np


def get_best_indices(ranks: np.ndarray, top: int, axis: int = 1) -> np.ndarray:
    """
    Returns indices of top largest values in rows of array.

    Parameters
    ----------
    ranks : np.ndarray, required
        Input array.
    top : int, required
        Number of largest values.
    axis : int, optional
        Axis along which the function is performed.
    """
    if axis == 1:
        part = np.argpartition(ranks, kth=-top, axis=1)[:, -top:, ...]
        return np.take_along_axis(part, np.argsort(np.take_along_axis(ranks, part, axis=1), axis=1)[:, ::-1], axis=1)
    sorted_array = np.argsort(np.swapaxes(ranks, 1, axis), axis=1)
    length = sorted_array.shape[1]
    return np.swapaxes(np.flip(sorted_array[:, length - top:, ...], 1), axis, 1)


if __name__ == "__main__":
    # здесь нужно считать файл
    with open('input.bin', 'rb') as f_data:
        ranks = np.load(f_data)
    # вызвать функцию get_best_indices
    top = 5
    axis = 1
    indices = get_best_indices(ranks, top, axis)
    # записать результат в файл
    with open('output.bin', 'wb') as f_data:
        np.save(f_data, indices)

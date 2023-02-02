"""
Задача C
replace_nan_to_means
Напишите функцию replace_nan_to_means(X), принимающую матрицу X. Функция должна вернуть копию матрицы X, в которой все
значения nan в каждом столбце заменены на среднее арифметическое остальных элементов столбца. В случае столбца из одних
nan необходимо заменить все элементы столбца на нули. Исходная матрица X должна остаться неизменной.
Функция должна быть написана с использованием библиотеки numpy. Использование циклов категорически запрещено.
Пример:
Входные данные
replace_nan_to_means(np.array([[0, 1, 2], [np.nan, 1, np.nan], [5, 6, 7]])
Результат работы
>>>np.array([[0, 1, 2], [2.5, 1, 4.5], [5, 6, 7]])
"""
import numpy as np


def replace_nan_to_means(X):
    is_nan_cols = np.isnan(X).sum(axis=0) == X.shape[0]
    means = np.zeros((X.shape[1]))
    means[~is_nan_cols] = np.nanmean(X[:, ~is_nan_cols], axis=0)
    M = np.copy(X)
    M[np.where(np.isnan(M))] = np.tile(means, (X.shape[0], 1))[np.where(np.isnan(M))]
    return M

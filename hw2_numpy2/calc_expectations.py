"""
Задача B
calc_expectations
Охотники за сокровищами отправились за очередным кладом в необычное место, которое называется "Поле чудес". "Поле чудес"
имеет размер H на W метров. На каждом квадратном метре "Поля чудес" закопан сундук с определённым количеством камней.
Скоро "Поле чудес" должен посетить лепрекон. Под влиянием лепрекона камни в сундуках превращаются в золотые монеты.
Лепрекон появляется в определённой клетке поля и действует на прямоугольную область размером h на w, верхний левый край
этой области - точка, где стоит лепрекон. Охотники за сокровищами не знают, когда точно появятся лепрекон, но хотят
получить как можно больше золота. Известно вероятностное распределение на появление лепрекона в каждый момент времени в
каждой клетке поля. Необходимо для каждой клетки "Поля чудес" посчитать математическое ожидание награды, которое будет
получено от этой клетки.
Необходимо написать функцию calc_expectations(h, w, X, Q), где:
h, w - размеры области влияния лепрекона;
X - матрица целых чисел размера H на W; X[i, j] - количество камней в i, j клетке;
Q - матрица вещественных чисел размера H на W, задающая вероятностное распределение; Q[i, j] - вероятность
появления лепрекона в точке i, j.
Функция возвращает матрицу E размера H на W; E[i, j] – математическое ожидание награды в i, j клетке.
Функция должна быть написана с использованием библиотеки numpy. Использование циклов категорически запрещено.
Пример:
Входные данные
h, w = 2, 2
X = np.asarray([
    [1, 1, 1, 1],
    [2, 2, 2, 2],
    [3, 3, 3, 3],
    [4, 4, 4, 4],
], dtype=int)
Q = np.asarray([
    [0.20, 0.  , 0.3 , 0.1 ],
    [0.10, 0.  , 0.2 , 0.  ],
    [0.05, 0.  , 0.  , 0.  ],
    [0.  , 0.  , 0.  , 0.05],
], dtype=float)
calc_expectations(h, w, X, Q)
Результат работы
>>>np.array([
   [0.2 , 0.2 , 0.3 , 0.4 ],
   [0.6 , 0.6 , 1.  , 1.2 ],
   [0.45, 0.45, 0.6 , 0.6 ],
   [0.2 , 0.2 , 0.  , 0.2 ],
])
"""
import numpy as np


def calc_expectations(h, w, X, Q):
    Q_h, Q_w = Q.shape
    wide_Q = np.zeros((Q_h + h - 1, Q_w + w - 1))
    wide_Q[-Q_h:, -Q_w:] = Q
    M = np.zeros((Q_h, Q_w))

    def sum_window(wide_Q, X, h, w, i, j):
        return np.sum(wide_Q[i: (i + h), j: (j + w)]) * X[i, j]
    range_Q_h = np.array(list(range(Q_h)) * Q_w)
    range_Q_w = np.sort(np.vstack((np.array(list(range(Q_w)) * Q_h))).ravel())
    excluded = ['wide_Q', 'X', 'h', 'w']
    signature = '(n,m),(k,l),(),(),(),()->()'
    M = np.vectorize(sum_window, excluded=excluded, signature=signature)(wide_Q, X, h, w, range_Q_h, range_Q_w)
    return M.reshape(Q_w, Q_h).T

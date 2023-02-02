import numpy as np


def euclidean_distance(x, y) -> np.ndarray:
    """
    :param x: — np.ndarray размера N × D;
    :param y: — np.ndarray размера M × D.
    :return: — np.ndarray размера N × M, каждый элемент которого — евклидово расстояние
    между соответствующей парой векторов из массивов X и Y.
    """
    n, d = x.shape
    m, d = y.shape
    return (np.tile(np.sum(x ** 2, axis=1).reshape((n, 1)), m) + np.tile(np.sum(y ** 2, axis=1),
                                                                         n).reshape((n, m)) - 2 * x.dot(y.T)) ** 0.5


def cosine_distance(x, y) -> np.ndarray:
    """
    :rtype: object
    :param x: — np.ndarray размера N × D;
    :param y: — np.ndarray размера M × D.
    :return: — np.ndarray размера N × M, каждый элемент которого — косинусное расстояние
    между соответствующей парой векторов из массивов X и Y
    """
    n, d = x.shape
    m, d = y.shape
    return 1 - x.dot(y.T) / (np.sum(y**2, axis=1) * np.sum(x**2, axis=1)[:, np.newaxis]) ** 0.5

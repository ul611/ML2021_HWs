import numpy as np

from knn.distances import euclidean_distance, cosine_distance


def get_best_ranks(ranks, top, return_distance=False):
    """
    Returns indices of top smallest values in rows of array.
    Parameters
    ----------
    ranks : np.ndarray, required
        Input array.
    top : int, required
        Number of smallest values.
    return_distance : bool, optional
        Return distances or do not.
    """
    if top >= ranks.shape[1]:
        indices = np.argsort(ranks)
    else:
        prt = np.argpartition(ranks, kth=top, axis=1)[:, :top, ...]
        indices = np.take_along_axis(prt, np.argsort(np.take_along_axis(ranks, prt, axis=1), axis=1), axis=1)
    if return_distance:
        return np.take_along_axis(ranks, indices, axis=1), indices
    return indices


class NearestNeighborsFinder:
    def __init__(self, n_neighbors, metric="euclidean"):
        self.n_neighbors = n_neighbors

        if metric == "euclidean":
            self._metric_func = euclidean_distance
        elif metric == "cosine":
            self._metric_func = cosine_distance
        else:
            raise ValueError("Metric is not supported", metric)
        self.metric = metric

    def fit(self, X, y=None):
        self._X = X
        return self

    def kneighbors(self, X, return_distance=False):
        """
        Метод производит поиск ближайших соседей.
        :param X: тестовая выборка размера M;
        :param return_distance:  булев флаг, нужно ли вернуть расстояния для объектов.
        :return: В случае return_distance=True возвращает кортеж
        (distances, indices) из двух np.ndarray размера M × K, где
        • distances[i, j] — расстояние от i-го объекта, до его j-го ближайшего соседа;
        • indices[i, j] — индекс ближайшего соседа из обучающей выборки до объекта с индексом i.
        Если return_distance=False, возвращается только второй из указанных массивов.
        """
        return get_best_ranks(self._metric_func(X, self._X), self.n_neighbors, return_distance=return_distance)

import numpy as np

from sklearn.neighbors import NearestNeighbors
from knn.nearest_neighbors import NearestNeighborsFinder


class KNNClassifier:
    EPS = 1e-5

    def __init__(self, n_neighbors, algorithm='my_own', metric='euclidean', weights='uniform'):
        if algorithm == 'my_own':
            finder = NearestNeighborsFinder(n_neighbors=n_neighbors, metric=metric)
        elif algorithm in ('brute', 'ball_tree', 'kd_tree',):
            finder = NearestNeighbors(n_neighbors=n_neighbors, algorithm=algorithm, metric=metric)
        else:
            raise ValueError("Algorithm is not supported", metric)

        if weights not in ('uniform', 'distance'):
            raise ValueError("Weighted algorithm is not supported", weights)

        self._finder = finder
        self._weights = weights

    def fit(self, X, y=None):
        self._finder.fit(X)
        self._labels = np.asarray(y)
        return self

    def _predict_precomputed(self, indices, distances):
        """
        Вспомогательный метод для предсказания меток классов объектов тестовой выборки.
        :param indices: массив индексов, np.ndarray размера M × K;
        :param distances: массив расстояний, np.ndarray размера M × K;
        :return: одномерный np.ndarray размера M, состоящий
        из предсказаний алгоритма (меток классов) для объектов тестовой выборки по заданным массивам
        расстояний и индексов.
        https://stackoverflow.com/a/56050427
        """
        size_label = self._labels.size
        size_test = indices.shape[0]
        if self._weights == 'uniform':
            return np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1,
                                       arr=np.take_along_axis(np.tile(self._labels, size_test
                                                                      ).reshape((size_test, size_label)),
                                                              indices, axis=1))
        else:
            w = 1 / (distances + self.EPS)
            return np.asarray(list(map(lambda x: np.bincount(x[0], weights=x[1]).argmax(),
                                       zip(np.take_along_axis(np.tile(self._labels, size_test
                                                                      ).reshape((size_test, size_label)),
                                                              indices, axis=1), w))))

    def kneighbors(self, X, return_distance=False):
        return self._finder.kneighbors(X, return_distance=return_distance)

    def predict(self, X):
        distances, indices = self.kneighbors(X, return_distance=True)
        return self._predict_precomputed(indices, distances)


class BatchedKNNClassifier(KNNClassifier):
    '''
    Нам нужен этот класс, потому что мы хотим поддержку обработки батчами
    в том числе для классов поиска соседей из sklearn
    '''

    def __init__(self, n_neighbors, algorithm='my_own', metric='euclidean', weights='uniform', batch_size=None):
        KNNClassifier.__init__(
            self,
            n_neighbors=n_neighbors,
            algorithm=algorithm,
            weights=weights,
            metric=metric,
        )
        self._batch_size = batch_size

    def kneighbors(self, X: object, return_distance: object = False) -> object:
        if self._batch_size is None or self._batch_size >= X.shape[0]:
            return super().kneighbors(X, return_distance=return_distance)
        if self._batch_size > 1200:
            self._batch_size = 1200
        batched_distances = np.zeros((X.shape[0], self._finder.n_neighbors))
        batched_indices = np.zeros((X.shape[0], self._finder.n_neighbors), dtype=np.int32)
        if return_distance:
            for i in range(0, X.shape[0], self._batch_size):
                batched_distances[i:i + self._batch_size, ...], batched_indices[i:i + self._batch_size, ...] = super(
                ).kneighbors(X[i:i + self._batch_size, ...], return_distance=return_distance)
            return batched_distances, batched_indices
        for i in range(0, X.shape[0], self._batch_size):
            batched_indices[i:i + self._batch_size, ...] = super().kneighbors(X[i:i + self._batch_size, ...],
                                                                              return_distance=return_distance)
        return batched_indices

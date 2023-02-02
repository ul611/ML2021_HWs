from collections import defaultdict

import numpy as np

from sklearn.model_selection import KFold, BaseCrossValidator
from sklearn.metrics import accuracy_score

from knn.classification import BatchedKNNClassifier


def knn_cross_val_score(X, y, k_list, scoring, cv=None, **kwargs):
    """
    Функция для применения кросс-валидации.
    :param X: обучающая выборка;
    :param y: ответы объектов на обучающей выборке;
    :param k_list:  список из проверяемых значений для числа ближайших соседей;
    :param scoring: название метрики, по которой оценивается качество алгоритма. Обязательно должна
    быть реализована метрика ’accuracy’ (доля правильно предсказанных ответов);
    :param cv: класс, реализующий интерфейс sklearn.model_selection.BaseCrossValidator для кроссвалидации,
    например, класс sklearn.model_selection.KFold.
    :param kwargs: параметры конструктора класса knn.classifier.KNNClassifier.
    :return: словарь, где ключами являются значения K из k_list, а элементами
    — np.ndarray размера len(cv) с качеством на каждом фолде.
    """
    y = np.asarray(y)

    if scoring == "accuracy":
        scorer = accuracy_score
    else:
        raise ValueError("Unknown scoring metric", scoring)

    if cv is None:
        cv = KFold(n_splits=5)
    elif not isinstance(cv, BaseCrossValidator):
        raise TypeError("cv should be BaseCrossValidator instance", type(cv))
    max_k = max(k_list)
    bclf = BatchedKNNClassifier(n_neighbors=max_k, **kwargs)
    k_accuracies = {}
    for train_index, test_index in cv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        bclf.fit(X_train, y_train)
        distances, indices = bclf.kneighbors(X_test, return_distance=True)
        for k in k_list:
            if k not in k_accuracies:
                k_accuracies[k] = np.array([])
            y_pred = bclf._predict_precomputed(indices[:, :k], distances[:, :k])
            k_accuracies[k] = np.hstack((k_accuracies[k], np.array([scorer(y_test, y_pred)])))
    return k_accuracies

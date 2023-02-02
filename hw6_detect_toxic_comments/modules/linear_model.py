import numpy as np
import scipy
from time import time
from sklearn.metrics import balanced_accuracy_score


class LinearModel:
    def __init__(
        self,
        loss_function,
        batch_size=None,
        step_alpha=1,
        step_beta=0,
        tolerance=1e-5,
        max_iter=1000,
        random_seed=153,
        **kwargs
    ):
        """
        Parameters
        ----------
        loss_function : BaseLoss inherited instance
            Loss function to use
        batch_size : int
        step_alpha : float
        step_beta : float
            step_alpha and step_beta define the learning rate behaviour
        tolerance : float
            Tolerace for stop criterio.
        max_iter : int
            Max amount of epoches in method.
        """
        self.loss_function = loss_function
        self.batch_size = batch_size
        self.step_alpha = step_alpha
        self.step_beta = step_beta
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.random_seed = random_seed

    def fit(self, X, y, w_0=None, trace=False, X_val=None, y_val=None):
        """

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse.csr_matrix
            2d matrix, training set.
        y : numpy.ndarray
            1d vector, target values.
        w_0 : numpy.ndarray
            1d vector in binary classification.
            2d matrix in multiclass classification.
            Initial approximation for SGD method.
        trace : bool
            If True need to calculate metrics on each iteration.
        X_val : numpy.ndarray or scipy.sparse.csr_matrix
            2d matrix, validation set.
        y_val: numpy.ndarray
            1d vector, target values for validation set.

        Returns
        -------
        : dict
            Keys are 'time', 'func', 'func_val'.
            Each key correspond to list of metric values after each training epoch.
        """
        # values for returning dictionnary
        ts, fs, fv = [], [], []

        nrows, ncols = X.shape

        # initialize variables
        np.random.seed(self.random_seed)
        self.batch_size = self.batch_size if self.batch_size else nrows
        w_0 = np.random.rand(ncols) if w_0 is None else w_0

        for k in range(self.max_iter):
            time_s = time()
            e_k = self.step_alpha / (k + 1) ** self.step_beta
            inds = np.random.permutation(nrows)

            for bs in range(0, nrows, self.batch_size):
                i = inds[bs:(bs + self.batch_size)]
                self.w_pred = w_0 - e_k * self.loss_function.grad(X[i, ...],
                                                                  y[i], w_0)
                dw = self.w_pred - w_0
                w_0 = self.w_pred

                if np.dot(dw, dw) <= self.tolerance:
                    if trace:
                        fs += [self.loss_function.func(X, y, w_0)]
                        fv += [X_val if X_val is None
                               else self.loss_function.func(X_val, y_val, w_0)]
                        time_e = time()
                        ts += [time_e - time_s]
                        return {'func':fs,
                                'func_val':fv,
                                'time':ts}
            if trace:
                fs += [self.loss_function.func(X, y, w_0)]
                fv += [X_val if X_val is None
                       else self.loss_function.func(X_val, y_val, w_0)]
                time_e = time()
                ts += [time_e - time_s]

        if trace:
            return {'func':fs,
                    'func_val':fv,
                    'time':ts}

    def predict(self, X, threshold=0):
        """

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse.csr_matrix
            2d matrix, test set.
        threshold : float
            Chosen target binarization threshold.

        Returns
        -------
        : numpy.ndarray
            answers on a test set
        """
        if type(X) == np.ndarray:
            proba = np.matmul(self.w_pred, X.T)
        else:
            proba = scipy.sparse.csr_matrix.dot(X, self.w_pred)
        y_pred = np.where(proba > threshold, 1, -1)
        return y_pred

    def get_optimal_threshold(self, X, y):
        """
        Get optimal target binarization threshold.
        Balanced accuracy metric is used.

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse.csr_matrix
            2d matrix, validation set.
        y : numpy.ndarray
            1d vector, target values for validation set.

        Returns
        -------
        : float
            Chosen threshold.
        """
        if self.loss_function.is_multiclass_task:
            raise TypeError('optimal threhold procedure is only for binary task')

        weights = self.get_weights()
        scores = X.dot(weights)
        y_to_index = {-1: 0, 1: 1}

        # for each score store real targets that correspond score
        score_to_y = dict()
        score_to_y[min(scores) - 1e-5] = [0, 0]
        for one_score, one_y in zip(scores, y):
            score_to_y.setdefault(one_score, [0, 0])
            score_to_y[one_score][y_to_index[one_y]] += 1

        # ith element of cum_sums is amount of y <= alpha
        scores, y_counts = zip(*sorted(score_to_y.items(), key=lambda x: x[0]))
        cum_sums = np.array(y_counts).cumsum(axis=0)

        # count balanced accuracy for each threshold
        recall_for_negative = cum_sums[:, 0] / cum_sums[-1][0]
        recall_for_positive = 1 - cum_sums[:, 1] / cum_sums[-1][1]
        ba_accuracy_values = 0.5 * (recall_for_positive + recall_for_negative)
        best_score = scores[np.argmax(ba_accuracy_values)]
        return best_score

    def get_weights(self):
        """
        Get model weights

        Returns
        -------
        : numpy.ndarray
            1d vector in binary classification.
            2d matrix in multiclass classification.
            Initial approximation for SGD method.
        """
        return self.w_pred

    def get_objective(self, X, y):
        """
        Get objective.

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse.csr_matrix
            2d matrix.
        y : numpy.ndarray
            1d vector, target values for X.

        Returns
        -------
        : float
        """
        return self.loss_function.func(X, y, self.w_pred)

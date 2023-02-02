import numpy as np


def get_numeric_grad(f, x, eps):
    """
    Function to calculate numeric gradient of f function in x.

    Parameters
    ----------
    f : callable
    x : numpy.ndarray
        1d array, function argument
    eps : float
        Tolerance

    Returns
    -------
    : numpy.ndarray
        Numeric gradient.
    """
    g = np.zeros(x.size)
    for i, x_i in enumerate(x):
        g[i] = abs(f(x_i + eps) - f(x_i)) / eps
    return g


def compute_balanced_accuracy(true_y, pred_y):
    """
    Get balaced accuracy value

    Parameters
    ----------
    true_y : numpy.ndarray
        True target.
    pred_y : numpy.ndarray
        Predictions.
    Returns
    -------
    : float
    """
    possible_y = set(true_y)
    value = 0
    for current_y in possible_y:
        mask = true_y == current_y
        value += (pred_y[mask] == current_y).sum() / mask.sum()
    return value / len(possible_y)

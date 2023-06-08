import numpy as np


def euclidean_distance(x, y):
    return np.sqrt(
        np.sum(x**2, axis=1).reshape(-1, 1)
        + np.sum(y**2, axis=1)
        - 2 * np.dot(x, y.T)
    )


def cosine_distance(x, y):
    num = np.dot(x, y.T)
    p1 = np.sqrt(np.sum(x**2, axis=1))[:, np.newaxis]
    p2 = np.sqrt(np.sum(y**2, axis=1))[np.newaxis, :]
    return 1 - num / (p1 * p2)

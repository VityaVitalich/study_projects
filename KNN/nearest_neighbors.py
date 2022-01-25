import numpy as np
from knn.distances import euclidean_distance, cosine_distance


def get_best_ranks(ranks, top, axis=1, return_ranks=False):
    if return_ranks:
        idx = np.argpartition(ranks, top-1, axis=1)[:, :top]
        sorte = np.take_along_axis(ranks, idx, axis=1)
        return (np.sort(sorte), np.take_along_axis(idx, np.argsort(sorte), axis=1))
    else:
        idx = np.argpartition(ranks, top-1, axis=1)[:, :top]
        sorte = np.take_along_axis(ranks, idx, axis=1)
        return np.take_along_axis(idx, np.argsort(sorte), axis=1)


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
        self.dists = self._metric_func(X, self._X)
        # num_train x num_test
        return get_best_ranks(self.dists, self.n_neighbors, return_ranks=return_distance)
        # k x num_test - номера объектов трейна

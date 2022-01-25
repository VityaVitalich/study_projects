import numpy as np
from sklearn.neighbors import NearestNeighbors
from knn.nearest_neighbors import NearestNeighborsFinder


class KNNClassifier:

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
        EPS = 1e-5
        self.w = 1 / (distances + EPS)

        def voting(arr):
            return np.argmax(np.bincount(arr))

        self.neighbors_labels = self._labels[indices]
        if self._weights == "uniform":
            self.voted_labels = np.apply_along_axis(voting, 1, self.neighbors_labels)
        else:
            ohl = (np.arange(self.neighbors_labels.max() + 1) == self.neighbors_labels[..., None]).astype(int)
            self.voted_labels = np.argmax(np.sum(ohl * self.w[:, :, np.newaxis], axis=1), axis=1)
        return self.voted_labels

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
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.weights = weights
        self.metric = metric

    def kneighbors(self, X, return_distance=False):
        if self._batch_size is None or self._batch_size >= X.shape[0]:
            return super().kneighbors(X, return_distance=return_distance)

        else:
            self.batches = X.shape[0] // self._batch_size
            self.num_obj = X.shape[1]
            X_batched = X[:self.batches * self._batch_size]
            X_tail = X[self.batches * self._batch_size:]
            self.X_new = np.split(X_batched, self.batches)
            self.X_new.append(X_tail)
            self.res = [0]*len(self.X_new)

            if return_distance:
                self.dist = [0]*len(self.X_new)
                for i, elem in enumerate(self.X_new):
                    self.tup = super().kneighbors(elem, return_distance=return_distance)
                    self.res[i] = self.tup[1]
                    self.dist[i] = self.tup[0]

                return (np.vstack(self.dist), np.vstack(self.res))

            else:
                for i, elem in enumerate(self.X_new):
                    self.tup = super().kneighbors(elem, return_distance=return_distance)
                    self.res[i] = self.tup

                return np.vstack(self.res)
    def params(self):
        return (self._n_neighbors, self.algorithm, self.weights, self.metric)
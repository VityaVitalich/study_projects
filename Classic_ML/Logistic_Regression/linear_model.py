import numpy as np
from scipy.special import expit
import time
from collections import defaultdict


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
        np.random.seed(random_seed)

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
        self.M = X.shape[1]
        self.N = X.shape[0]
        if w_0 is None:
            self.w = np.random.normal(0, 0.01, self.M)
        else:
            self.w = w_0

        if trace:
            history = defaultdict()
            history["func"] = []
            history["time"] = []
            if (X_val is not None) and (y_val is not None):
                history["func_val"] = []

                f_val = self.loss_function.func(X_val, y_val, self.w)
                history["func_val"].append(f_val)

            history["time"].append(0)
            f = self.loss_function.func(X, y, self.w)
            history["func"].append(f)

        for k in range(self.max_iter):
            start = time.time()
            n_k = self.step_alpha / (k + 1) ** self.step_beta
            w_t = self.w

            if self.batch_size is not None:
                idx = np.arange(self.N)
                np.random.shuffle(idx)
                batches = np.ceil(self.N / self.batch_size)
                idx_ls = np.array_split(idx, batches)
                for batch_idx in idx_ls:
                    X_batched = X[batch_idx]
                    y_batched = y[batch_idx]
                    grad = self.loss_function.grad(X_batched, y_batched, self.w)
                    self.w = self.w - n_k * grad
            else:
                X_batched = X
                y_batched = y
                grad = self.loss_function.grad(X_batched, y_batched, self.w)
                self.w = self.w - n_k * grad

            end = time.time()
            stop_criterion = np.linalg.norm(self.w - w_t)

            if trace:
                time_spent = end - start
                history["time"].append(time_spent)
                functional_train = self.loss_function.func(X, y, self.w)
                history["func"].append(functional_train)

                if (X_val is not None) and (y_val is not None):
                    functional_val = self.loss_function.func(X_val, y_val, self.w)
                    history["func_val"].append(functional_val)

            if stop_criterion <= self.tolerance:
                break

        if trace:
            return history

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

        return np.sign((X @ self.w) - threshold)

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
            raise TypeError("optimal threhold procedure is only for binary task")

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
        return self.w

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
        return self.loss_function.func(X, y, self.w)

    def compute_balanced_accuracy(self, true_y, pred_y):
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

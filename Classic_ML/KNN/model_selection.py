from collections import defaultdict

import numpy as np

from sklearn.model_selection import KFold, BaseCrossValidator
from sklearn.metrics import accuracy_score

from knn.classification import BatchedKNNClassifier


def knn_cross_val_score(X, y, k_list, scoring, cv=None, **kwargs):
    y = np.asarray(y)

    if scoring == "accuracy":
        scorer = accuracy_score
    else:
        raise ValueError("Unknown scoring metric", scoring)
        
        
    if cv is None:
        cv = KFold(n_splits=5)
    elif not isinstance(cv, BaseCrossValidator):
        raise TypeError("cv should be BaseCrossValidator instance", type(cv))

    k_dict = defaultdict(float)
    for tr_ind, test_ind in cv.split(X):
        x_train = X[tr_ind]
        x_test = X[test_ind]
        y_train = y[tr_ind]
        y_test = y[test_ind]
        maxk = np.max(k_list)
        cl = BatchedKNNClassifier(maxk, **kwargs)
        cl.fit(x_train, y_train)

        max_dist, max_ind = cl.kneighbors(x_test, return_distance=True)
        for k in k_list:
            if (k == maxk):
                y_pred = cl._predict_precomputed(max_ind, max_dist)
            else:
                y_pred = cl._predict_precomputed(max_ind[:, :k], max_dist[:, :k])

            acc = accuracy_score(y_pred, y_test)

            if k not in k_dict.keys():
                k_dict[k] = np.array(acc)
            else:
                curr_score = k_dict[k]
                curr_score = np.append(curr_score, acc)
                k_dict[k] = curr_score

    return k_dict

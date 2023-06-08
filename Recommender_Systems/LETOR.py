from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import numpy as np
from tqdm import tqdm


class LambdaMart:
    def __init__(
        self,
        n_trees=10,
        max_leaf_nodes=10,
        lr=0.01,
        max_depth=None,
        min_samples_split=2,
        max_rank=1,
    ):
        self.n_trees = n_trees
        self.mln = max_leaf_nodes
        self.lr = lr
        self.md = max_depth
        self.mss = min_samples_split
        self.params = {
            "min_samples_split": min_samples_split,
            "max_depth": max_depth,
            "max_leaf_nodes": max_leaf_nodes,
        }

        self.max_rank = max_rank

    def fit(self, X_train, verbose=True, ndcg=10):
        self.X = X_train.copy()
        self.X["prev_res"] = 0.0

        self.ensemble = []

        for i in range(self.n_trees):
            self.X = self.transform_df(self.X)

            X = self.X.filter(regex="feature")
            y = self.X["lambda"]

            model = DecisionTreeRegressor(**self.params)
            model.fit(X, y)

            self.ensemble.append(model)

            curr_pred = model.predict(X)
            self.X["prev_res"] += curr_pred * self.lr

            if verbose:
                metric = self.compute_metric(self.X, ndcg)
                print("{} iteration nDCGs == {} ".format(i, metric))

            self.X = self.X.drop(["lambda"], axis=1)

    def predict(self, X_val):
        self.X_val = X_val.copy()
        self.X_val["prev_res"] = 0.0

        for i in range(self.n_trees):
            self.X_val = self.transform_df(self.X_val)

            X = self.X_val.filter(regex="feature")

            curr_pred = self.ensemble[i].predict(X)
            self.X_val["prev_res"] += curr_pred * self.lr

            self.X_val = self.X_val.drop(["lambda"], axis=1)

        return self.X_val

    def compute_metric(self, X, ndcg):
        X["DCG"] = X["gain"] * X["dc"]
        X["IDCG"] = X["Igain"] * X["dc"]
        dcg = X[X["rank_sorted"] < ndcg].groupby("query_id")["DCG"].sum().mean()
        idcg = X[X["rank_sorted"] < ndcg].groupby("query_id")["IDCG"].sum().mean()

        return dcg / idcg

    def transform_df(self, X):
        X = X.sort_values(["query_id", "prev_res"], ascending=[True, False])
        X["rank_sorted"] = X.groupby("query_id").cumcount()

        X["dc"] = 1 / np.log2(2 + X["rank_sorted"])
        X["gain"] = X["relevance"]

        X["Igain"] = np.ones_like(X["relevance"]) * self.max_rank

        lambdas = self.compute_swaps_lambda(X)

        X = X.merge(
            lambdas,
            left_on=["query_id", "rank_sorted"],
            right_on=["query_id", "rank_sorted_x"],
            how="left",
        )

        return X

    def compute_swaps_lambda(self, X):
        swaps = X.merge(X, on="query_id", how="outer")

        swaps["diff"] = np.abs(
            (swaps["gain_x"] - swaps["gain_y"]) * (swaps["dc_x"] - swaps["dc_y"])
        )
        swaps["lambda"] = 0

        good_swap = swaps[swaps["relevance_x"] > swaps["relevance_y"]]
        swaps.loc[swaps["relevance_x"] > swaps["relevance_y"], "lambda"] = good_swap[
            "diff"
        ]

        lambdas = (
            swaps.groupby(["query_id", "rank_sorted_x"])["lambda"].sum()
            - swaps.groupby(["query_id", "rank_sorted_y"])["lambda"].sum()
        )

        return lambdas


def nDCG(rel_vect, n, max_rank=1):
    s = 0
    iDCG = 0
    if n > len(rel_vect):
        n = len(rel_vect)

    for k in range(1, n + 1):
        s += rel_vect[k - 1] / (np.log2(k + 1))
        iDCG += max_rank / (np.log2(k + 1))

    return s / iDCG


def mean_nDCG(df, n, max_rank=1):
    querys = df["query_id"].unique()
    avg_ndcg_ls = []
    for q in querys:
        relevance_vector = (
            df[(df["query_id"] == q)]
            .sort_values(by="prev_res", ascending=False)["relevance"]
            .values
        )

        avg_ndcg = nDCG(relevance_vector, n, max_rank)
        avg_ndcg_ls.append(avg_ndcg)

    return np.mean(avg_ndcg_ls)


def MRR(df):
    querys = df["query_id"].unique()
    rec_rank_ls = []
    for q in querys:
        relevance_vector = (
            df[(df["query_id"] == q)]
            .sort_values(by="prev_res", ascending=False)["relevance"]
            .values
        )
        try:
            id_first_relevant = np.where(relevance_vector != 0)[0][0]
            # n = len(relevance_vector)
            reciprokal_rank = 1 / (id_first_relevant + 1)
        except IndexError:
            reciprokal_rank = 0

        rec_rank_ls.append(reciprokal_rank)

    return np.mean(rec_rank_ls)


def precision_at_k(rel_vect, k):
    s_retr = len(rel_vect[:k].nonzero()[0])
    # print(s_retr)
    s_rel = k

    return s_retr / s_rel


def avg_precision_at_n(rel_vect, n):
    s = 0
    if n > len(rel_vect):
        n = len(rel_vect)

    for k in range(1, n + 1):
        if rel_vect[k - 1] > 0:
            rel = 1
        else:
            rel = 0
        prec_at_k = precision_at_k(rel_vect, k) * rel
        s += prec_at_k

    return s / n


def MAP(df, n):
    querys = df["query_id"].unique()
    avg_prec_ls = []
    for q in querys:
        relevance_vector = (
            df[(df["query_id"] == q)]
            .sort_values(by="prev_res", ascending=False)["relevance"]
            .values
        )

        avg_prec = avg_precision_at_n(relevance_vector, n)
        avg_prec_ls.append(avg_prec)

    return np.mean(avg_prec_ls)

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import pairwise_distances


class KNN:
    def __init__(self, k =5, metric='cosine', metric_func=None):
        self.k = k
        self.metric = metric
        self.metric_func = metric_func
        
    def fit(self, X):
        '''
        X - dataframe with zeros where no ratings
        '''
        self.cols = X.columns
        self.idx_df = X.index
        self.X = X.to_numpy()
        
        self.matrix_filled = False
        

        self.sim_matrix = self.compute_similarity(self.X, self.metric, self.metric_func)

        self.n = self.get_best_ranks(self.sim_matrix, self.k)

    def get_mean_res(self, neighbors_idx):

        ls_neighbors_values = []
        for idx in neighbors_idx:
            out = self.X[idx][np.newaxis,:]
            ls_neighbors_values.append(out)

        res = np.concatenate(ls_neighbors_values, axis=0)

        num_non_zero = np.count_nonzero(res, axis=0)

        mean_ratings = np.sum(res,axis=0)/num_non_zero

        return mean_ratings


    def predict_row(self, idx):

        mask = (self.X[idx] == 0).astype(int)

        neighbors = self.n[idx]

        mean_res = self.get_mean_res(neighbors)

        return self.X[idx]+ mask*mean_res

    def fill_matrix(self):
        self.Y = np.array(self.X)
        for i in (range(len(self.X))):
            self.Y[i] = self.predict_row(i)
            

        self.y_df = pd.DataFrame(self.Y)

        means_columns = self.y_df.mean(axis=0)
        self.y_df = self.y_df.fillna(means_columns)

        self.y_df.columns = self.cols
        self.y_df.index = self.idx_df

        self.matrix_filled = True
        

    def predict(self, X_test):

        if not self.matrix_filled:
            self.fill_matrix()

        y_test = []
        y_pred = []

        for idx, row in (X_test.iterrows()):

            user = row['user']
            item = row['item']
            rating = row['rating']

            y_test.append(rating)
            try:
                y_pred.append(self.y_df.loc[item][user])
            except KeyError:
                y_pred.append(0)

        self.y_pred = np.array(y_pred)
        self.y_test = np.array(y_test)

        return self.y_pred, self.y_test

        
    

    @staticmethod
    def get_best_ranks(ranks, top, axis=1, return_ranks=False):
        if return_ranks:
            idx = np.argpartition(ranks, top, axis=1)[:, :top+1]
            sorte = np.take_along_axis(ranks, idx, axis=1)
            return (np.sort(sorte)[:,1:], np.take_along_axis(idx, np.argsort(sorte), axis=1)[:,1:])
        else:
            idx = np.argpartition(ranks, top, axis=1)[:, :top+1]
            sorte = np.take_along_axis(ranks, idx, axis=1)
            return np.take_along_axis(idx, np.argsort(sorte), axis=1)[:,1:]

    @staticmethod
    def compute_similarity(X, metric='cosine', computing_function=None):

        if metric in ['cosine', 'euclidean', 'manhattan']:
            sim = pairwise_distances(X, metric=metric)
        else:
            sim = computing_function(X)
                

        return sim


class KNNBasic(KNN):
    def __init__(self, k = 5, metric='cosine', metric_func=None):
        super().__init__(k, metric, metric_func)

    
class KNNMeans(KNN):
    def __init__(self, k = 5, metric='cosine', metric_func=None):
        super().__init__(k, metric, metric_func)
        
    def fit(self, X):
        '''
        X - dataframe with zeros where no ratings
        '''
        self.cols = X.columns
        self.idx_df = X.index
        self.X = X.to_numpy()
        
        self.matrix_filled = False

        self.row_mean = np.true_divide(X.sum(1),(X!=0).sum(1))
        

        self.sim_matrix = self.compute_similarity(self.X)

        self.n = self.get_best_ranks(self.sim_matrix, self.k, return_ranks=True)


    def get_mean_res(self, neighbors, dists):

        ls_neighbors_values = []
        norm_coef_ls = []

        for i in range(len(neighbors)):
            dist = dists[i]
            idx = neighbors[i]

            non_zero_mask = self.X[idx] != 0 
            norming_coef = non_zero_mask*dist

            row_mean_idx = non_zero_mask * self.row_mean[idx]


            out = dist * (self.X[idx][np.newaxis,:] - row_mean_idx)


            norm_coef_ls.append(norming_coef)
            ls_neighbors_values.append(out)

        res = np.concatenate(ls_neighbors_values, axis=0)

        denominator = np.sum(norm_coef_ls, axis=0)

        mean_ratings = self.row_mean[idx] + np.sum(res,axis=0)/denominator

        return mean_ratings



    def predict_row(self, idx):

        mask = (self.X[idx] == 0).astype(int)

        neighbors = self.n[1][idx]
        dists = self.n[0][idx]

        mean_res = self.get_mean_res(neighbors, dists)

        return self.X[idx]+ mask*mean_res

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import mean_squared_error



class SVD():

    def __init__(self, n_factors=5, mean_norm=0, std_norm=1):
        self.n_factors = n_factors
        self.mean = mean_norm
        self.std = std_norm


    def fit(self, X, n_iters=1000, lr_pu=0.01, lr_qi=0.01, lr_bu=0.01, lr_bi=0.01, lr_y=0.1, raw_data=True, verbose=10000, algo='sgd', n_epoch=5):

        '''
        X - dataframe user - item - ratings
        '''

        if raw_data:
            self.encode_df(X)
        else:
            self.X = X


        self.n_epoch = n_epoch
        self.lr_y = lr_y
        self.lr_pu = lr_pu
        self.lr_qi = lr_qi
        self.lr_bu = lr_bu
        self.lr_bi = lr_bi
        self.n_users = int(self.X['user'].max() + 1)
        self.n_items = int(self.X['item'].max() + 1)
        self.global_mean = self.X['rating'].mean()
        self.n_iters = n_iters
        self.verbose = verbose
        self.algo = algo

        self.init_weights()

 


        if algo == 'sgd':
            self.fit_sgd()
        elif algo == 'als':
            self.fit_als()
        elif algo == 'svd++':
            self.fit_svd_pp()
        else:
            raise ValueError('No algo')


    def fit_sgd(self):
        self.sample = self.X.sample(self.n_iters)
        
        k = 0
        mean_err = 0
        for idx, row in self.sample.iterrows():
            k+= 1
            u = int(row['user'])
            i = int(row['item'])
            r = row['rating']

            err = r - (self.global_mean + self.bu[u] + self.bi[i] + self.Q[i]@self.P[u].T)
            mean_err += err

            if self.verbose is not None:
                if k%self.verbose == 0:
                    print(mean_err/k)


            self.bu[u] += self.lr_bu * err - self.lr_bu * self.bu[u]
            self.bi[i] += self.lr_bi * err - self.lr_bi * self.bi[i]

            self.Q[i] = self.lr_qi * err * self.Q[i]
            self.P[u] = self.lr_pu * err * self.Q[u]

        print(mean_err/k)

    def fit_svd_pp(self):
        self.sample = self.X.sample(self.n_iters)

        k = 0
        mean_err = 0
        for idx, row in self.sample.iterrows():
            k += 1
            u = int(row['user'])
            i = int(row['item'])
            r = row['rating']

            Vu = (self.X[(self.X['user'] == u)]['item'].values).astype(int)

            sum_yuj = (self.y[Vu,:].sum(axis=0) / np.sqrt(len(Vu)))

            err = r - (self.global_mean + self.bu[u] + self.bi[i] + self.Q[i]@(self.P[u].T + sum_yuj))
            mean_err += err

            if self.verbose is not None:
                if k%self.verbose == 0:
                    print(mean_err/k)


            self.bu[u] += self.lr_bu * err - self.lr_bu * self.bu[u]
            self.bi[i] += self.lr_bi * err - self.lr_bi * self.bi[i]

            self.Q[i] = self.lr_qi * err * self.Q[i]
            self.P[u] = self.lr_pu * err * (self.P[u] + sum_yuj)
            self.y[Vu,:] += self.lr_y * (err * self.Q[i]/np.sqrt(len(Vu)))


        print(mean_err/k)

    def fit_als(self):

        for i in range(self.n_epoch):
            
            self.update_Q()
 
            self.update_P()


            if self.verbose is not None:
                sample = self.X.sample(self.verbose)
                y_true = []
                y_pred = []
                for idx, row in sample.iterrows():
                    u = int(row['user'])
                    i = int(row['item'])
                    r = row['rating']

                    y_true.append(r)
                    y_pred.append(self.Q[i]@self.P[u].T)

                print(mean_squared_error(y_true, y_pred, squared = False))

    def update_Q(self):
        for i in range(self.n_items):
            sum_rui_p = 0
            sum_p = 0
            sample = self.X[(self.X['item'] == i)]
            for idx, row in sample.iterrows():
                u = int(row['user'])
                i = int(row['item'])
                r = row['rating']


                sum_rui_p += r*self.P[u]
                sum_p += np.outer(self.P[u], self.P[u].T)
            try:
                self.Q[i] = np.linalg.inv(sum_p)@sum_rui_p
            except np.linalg.LinAlgError:
                continue

    def update_P(self):
        for i in range(self.n_users):
            sum_rui_q = 0
            sum_q = 0
            sample = self.X[(self.X['user'] == i)]
            for idx, row in sample.iterrows():
                u = int(row['user'])
                i = int(row['item'])
                r = row['rating']


                sum_rui_q += r*self.Q[i]
                sum_q += np.outer(self.Q[i], self.Q[i].T)
            try:
                self.P[u] = np.linalg.inv(sum_q)@sum_rui_q
            except np.linalg.LinAlgError:
                continue
        

    def predict(self, X_test):

        self.transform_test(X_test)


        y_pred = []
        y_test = []

        for idx, row in self.X_test.iterrows():
            u = int(row['user'])
            i = int(row['item'])
            r = row['rating']

            y_test.append(r)

            if (u == -1) or (i == -1):
                out = self.global_mean
            else:
                if self.algo == 'sgd':
                    out = (self.global_mean + self.bu[u] + self.bi[i] + self.Q[i]@self.P[u].T)
                elif self.algo == 'als':
                    out = self.Q[i]@self.P[u].T
                elif self.algo == 'svd++':
                    Vu = (self.X[(self.X['user'] == u)]['item'].values).astype(int)

                    sum_yuj = (self.y[Vu,:].sum(axis=0) / np.sqrt(len(Vu)))
                    out = (self.global_mean + self.bu[u] + self.bi[i] + self.Q[i]@(self.P[u].T + sum_yuj))

            y_pred.append(out)

        y_pred = np.array(y_pred)
        y_pred = y_pred.clip(0, 5)

        y_test = np.array(y_test)

        return y_pred, y_test


    def init_weights(self):
        self.bu = np.zeros(self.n_users)
        self.bi = np.zeros(self.n_items)

        self.P = np.random.normal(self.mean, self.std, size=(self.n_users, self.n_factors))
        self.Q = np.random.normal(self.mean, self.std, size=(self.n_items, self.n_factors))

        self.y = np.random.normal(self.mean, self.std, size=(self.n_items, self.n_factors))



    def encode_df(self, X):
        self.le_users = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        self.le_items = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
        self.X = pd.DataFrame(X)

        self.X['user'] = self.le_users.fit_transform(self.X['user'].values.reshape(-1,1))
        self.X['item'] = self.le_items.fit_transform(self.X['item'].values.reshape(-1,1))


    def transform_test(self, X_test):

        self.X_test = pd.DataFrame(X_test)
        self.X_test['item'] = self.le_items.transform(self.X_test['item'].values.reshape(-1,1))
        self.X_test['user'] = self.le_users.transform(self.X_test['user'].values.reshape(-1,1))

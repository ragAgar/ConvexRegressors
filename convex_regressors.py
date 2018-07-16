#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from scipy.spatial import ConvexHull
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression


def mat(X):
    return X.reshape(len(X), -1)

def MAPE(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100       

def _adjusted_k_idx(k):
    new_k = k + 1
    #new_k = np.arange(len(np.unique(new_k)) )
    for new_k_idx in np.unique(new_k):
        np.put(new_k, [np.arange(len(new_k))[new_k == new_k_idx]], -new_k_idx)    
    _, indexs = np.unique(k,return_index=True)
    for i, new_k_idx in enumerate(np.sort(indexs)):
        np.put(new_k, [np.arange(len(new_k))[new_k == new_k[new_k_idx]]], i)    
    return new_k
    
#指定したX, yでlinear model fitしてlossを返す
def _get_lmfit_score(X, y):
    lm = LinearRegression()
    lm.fit(X.reshape(len(X), -1),  y.reshape(len(X), -1))
    y_pred = lm.predict(X.reshape(len(X), -1))
    return mean_squared_error(y_pred = y_pred, y_true = y), lm

#####################################################
#CAP Regressor

class CAPRegressor():
    def __init__(self, convex=True, max_iter  =1000):
        self.convex = convex
        self.max_iter = max_iter
        
    def fit(self, X, y, alpha_num = 10, D = 3):
        self.alpha_num = alpha_num        
        self.n_min = int(len(X)/(D*np.log(len(X))))
        self.C = np.arange(len(X))

        k = np.zeros(len(X))
        copy_X = X.copy()
        copy_y = y.copy()
        
        k = self._split_k(k, copy_X, copy_y, not_update= [])
        new_k = self._reupdate_k(k, copy_X, copy_y)
        lm_models, y_preds = self._cap_fit(X = copy_X, y = copy_y, k = new_k)
        self.lm_models = lm_models
        self.k  = new_k
        self.y_preds = y_preds

    def predict(self, X):
        # if self.convex:
        return self._cap_predict(X.reshape(len(X), -1), self.lm_models)
        #else:
            #return -self._cap_predict(X.reshape(len(X), -1), self.lm_models)

    #splitに使用する割合
    def _get_alpha_list(self):
        return np.arange(0, 1 , 1/self.alpha_num)

    #ratioによって、データセットをratioで分割する
    def _split_by_ratio(self, x, y, ratio):
        split_idx = self._get_ratio_idx(x, ratio)
        return x[:split_idx], x[split_idx:], y[:split_idx], y[split_idx:]

    #指定したratioのindex番号を返す
    def _get_ratio_idx(self, x, ratio):
        return int(len(x) * ratio)


    #splitする場合複数でるため、それを比率でかけて合計するもの
    def _get_balanced_score(self, score1, score2, x1, x2):
        return (score1*len(x1) + score2*len(x2)) / (len(x1) + len(x2))

    # alpha_listの値ごとに区切って全てのscoreを出して、一番良いalphaを返す
    def _get_best_split_score(self, X, y):
        score0, model0 = _get_lmfit_score(X, y) #splitする前の指定した区間でのfit
        alpha_list = self._get_alpha_list()    
        score_list = [score0]
        #from IPython.core.debugger import Pdb; Pdb().set_trace()

        for alpha in alpha_list[1:]:
            #from IPython.core.debugger import Pdb; Pdb().set_trace()
            if np.min([len(X)*alpha, len(X)*(1-alpha)]) > self.n_min: #サンプルサイズがn_minより上かどうか見る
                X_1, X_2, y_1, y_2 = self._split_by_ratio(X, y, alpha)
                score1, _ = _get_lmfit_score(X_1, y_1)
                score2, _ = _get_lmfit_score(X_2, y_2)
                score_list.append(self._get_balanced_score(score1, score2, X_1, X_2))
                
            else: #足りない場合, infを代入
                score_list.append(np.inf)

        updated = False if np.min(score_list) == score0 else True #更新しているかどうか
        return alpha_list[np.argmin(score_list)], updated

    #main, splitを繰り返す.
    def _update_k(self, k_before, X, y, not_update):
        k = k_before.copy()
        for k_idx in np.unique(k): 
            if k_idx not in not_update:
                best_ratio, updated = self._get_best_split_score(X[k == k_idx], y[k == k_idx])

                if updated:
                    best_ratio_idx = self._get_ratio_idx(k[k == k_idx], best_ratio)
                    C_k_idx = self.C[k ==k_idx]
                    np.put(k, C_k_idx[best_ratio_idx:], [len(np.unique(k))])
                    #np.put(k, C_k_idx[best_ratio_idx:], [len(np.unique(k))]) 
                else:
                    not_update.append(k_idx)
        return k, not_update
        #k[k == k_idx] = np.where(np.arange(len(k[k == k_idx])) < best_ratio_idx, k_idx + "-1", k_idx + "-2")#np.asarray([val + "-0" if i < best_ratio_idx else val + "-1" for i,val in enumerate(k[k == k_idx])])
        
    #実行部分
    def _split_k(self, k, X, y, not_update=[]):
        k_before = k.copy()
        while True:
            k_after, not_update = self._update_k(k_before, X, y, not_update)     
            _, counts = np.unique(k_after, return_counts=True)     
            if np.min(counts) < self.n_min:
                k_after = np.zeros(len(k_after))
                print("Error")
            if len(not_update) == len(np.unique(k_after)):
                return k_after
            k_before = k_after
            
    def _cap_fit(self, X, y, k):
        lm_models = []
        y_preds = []    
        for k_idx in np.unique(k):
            _,lm  = _get_lmfit_score(X[k == k_idx], y[k == k_idx])
            lm_models.append(lm)
            y_preds.append(np.ravel(lm.predict(X.reshape(len(X),-1))))
        return lm_models, np.asarray(y_preds)

    def _cap_predict(self, X, lm_models):
        y_preds = []
        for lm in lm_models:
            y_preds.append(np.ravel(lm.predict(X.reshape(len(X),-1))))
        if self.convex:
            return np.max(np.asarray(y_preds), axis=0)      
        else:
            return np.min(np.asarray(y_preds), axis=0) 

    def _refit_k(self, y_preds, k):
        new_k = np.argmax(y_preds, axis=0)
        new_k = _adjusted_k_idx(new_k)
        return new_k

    #サンプル数が不足しているnew_kのindexは元のkにする.
    
    def _get_new_k(self, y_preds, k):
        new_k= self._refit_k(y_preds, k)
        unique_k, counts = np.unique(new_k, return_counts=True)
         
        for i in range(len(unique_k)):
            unique_k, counts = np.unique(new_k, return_counts=True)
            if len(unique_k[counts < self.n_min]):
                for new_k_idx in unique_k[counts < self.n_min]:
                    try:
                        short_k_idx = np.arange(len(new_k))[new_k == new_k_idx][0]
                        short_k_idx = np.arange(len(k))[k == k[short_k_idx]]
                        np.put(new_k, [short_k_idx], new_k[short_k_idx[0]])
                    except:
                        pass
        return _adjusted_k_idx(new_k)

    def _reupdate_k(self, k, X, y):
        lm_models, y_preds = self._cap_fit(X = X, y = y, k =k)
        
        best_score = mean_squared_error(y_pred=self._cap_predict(X, lm_models), y_true=y)
        best_k = k.copy()
        
        for i in range(self.max_iter):
            new_k = self._get_new_k(y_preds, k)     
            k_idx,counts = np.unique(new_k, return_counts=True)
        
            if np.max(counts) >= 2*self.n_min:

                new_k = self._split_k(new_k, X, y, list(k_idx[counts < 2*self.n_min]))
                if np.sum(k == new_k) != len(k):
                    k = new_k
                    lm_models, y_preds = self._cap_fit(X = X, y = y, k =k)
                    error_score = mean_squared_error(y_pred=self._cap_predict(X, lm_models), y_true=y)
                    if error_score < best_score:
                        best_k = k.copy()
                        best_score = error_score
                else:
                    return new_k
                
            else:
                return new_k
        else: #forのelse
            return best_k

#####################################################

class ConvexPiecewiseLinearRegressor():
    def __init__(self, convex, max_iter=100):
        self.convex = convex
        self.max_iter = max_iter  
        
    def fit(self, X, y, num_k=10):
        init_p = self._get_init_p(X, num_k)
        k = self._get_min_dist_index(X, init_p)
        self.k = self._update_k(X, y, k)
        
    def predict(self, X):
        self.preds = []
        for reg in self.regs:
            self.preds.append(reg.predict(mat(X)).ravel())
        if self.convex:
            return np.max(self.preds, axis=0)
        else:
            return np.min(self.preds, axis=0)            
        
    def _get_init_p(self, X, num_k):    
        return X[np.random.choice(np.arange(len(X)), num_k, replace=False)]

    def _get_min_dist_index(self, X, vals):
        dist = []
        if X.ndim == 1:
            for i, val in enumerate(vals):
                dist.append((X - val)**2)    
        else:
            for i, val in enumerate(vals):
                dist.append(np.mean((X- val)**2, axis=1))    
        return _adjusted_k_idx(np.argmin(dist, axis=0))


    def _lm_fit(self, X, y, k):
        self.regs = []
        for k_idx in np.unique(k):
            X_j = X[k == k_idx]
            y_j = y[k == k_idx]
            _, lm_j = _get_lmfit_score(X_j, y_j)
            self.regs.append(lm_j)
        return self.regs

    def _lm_pred(self, X):
        self.preds = []
        for reg in self.regs:
            self.preds.append(reg.predict(mat(X)).ravel())
        if self.convex:
            return np.argmax(self.preds, axis=0)
        else:
            return np.argmin(self.preds, axis=0)

    def _update_k(self, X, y, k): 
        for i in range(self.max_iter):     
            regs = self._lm_fit(X, y, k)
            new_k = self._lm_pred(X)
            if np.sum(new_k != k) == 0:
                return _adjusted_k_idx(k)
            else:
                k = _adjusted_k_idx(new_k)
        else:
            return k
        
#####################################################


class ConvexifyRegressor():
    def __init__(self, convex=False):
        self.convex = convex
    
    def fit(self, X, y, trained_regressor, percentile = 5):
        y_pred = trained_regressor.predict(mat(X))
    
        is_in_M = self._get_betterfit_samples(y_pred, y, percentile)
        betterfit_points = np.asarray([X[is_in_M],y[is_in_M]]).T
        hull = ConvexHull(betterfit_points)
    
        regs = self._get_candidate_regs(hull, betterfit_points)
        self.regs = self._filter_regs(regs=regs , X = X,  y=y, is_in_M=is_in_M)
        
    def _get_candidate_regs(self, hull, points):
        regs = []
        for simplex in hull.simplices:
            X_hull = points[simplex, :-1]
            y_hull = points[simplex, -1]
            reg = LinearRegression()
            reg.fit(X = mat(X_hull) ,y = mat(y_hull))
            regs.append(reg)
        return regs        
    
    def _filter_regs(self, regs, X, y, is_in_M):
        preds = []
        regs = np.asarray(regs)
        for reg in regs:
            preds.append(reg.predict(mat(X[is_in_M])).ravel())
        if self.convex:
            return regs[np.sum(np.asarray(preds) <= y[is_in_M], axis=1) > 2]
        else:
            return regs[np.sum(np.asarray(preds) >= y[is_in_M], axis=1) > 2]

    def predict(self, X):
        preds = []
        for reg in self.regs:
            preds.append(reg.predict(mat(X)).ravel())
        if self.convex:
            return np.max(preds, axis=0)
        else:
            return np.min(preds, axis=0)  

    def _get_betterfit_samples(self, y_pred, y_true, percentile = 5):
        error = (y_pred-y_true)**2
        sigma = np.percentile(error, percentile)
        return error <= sigma      
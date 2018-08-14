"""
Implementation of k-nearest neighbours classifier
"""

import numpy as np
from scipy import stats
import utils

class KNN:

    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X # just memorize the trianing data
        self.y = y 

    def predict(self, Xtest):
        X, k, y = self.X, self.k, self.y
        D = utils.euclidean_dist_squared(X,Xtest)
        D_sorted = D.argsort(axis=0)

        n = len(D[0,:])
        yhat = np.zeros(n)

        for i in range(0, n):
            candidates = y[D_sorted[0:k,i]]
            yhat[i] = utils.mode(candidates)
        return yhat

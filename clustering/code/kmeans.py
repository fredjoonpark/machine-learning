import numpy as np
import math
from utils import euclidean_dist_squared

class Kmeans:

    def __init__(self, k):
        self.k = k

    def fit(self, X):
        N, D = X.shape
        y = np.ones(N)

        # Getting 500 random samples to be the mean
        means = np.zeros((self.k, D))
        for kk in range(self.k):
            i = np.random.randint(N)
            means[kk] = X[i]

            #self.means = means
            #print(self.error(X))

        while True:
            y_old = y

            # Compute euclidean distance to each mean
            dist2 = euclidean_dist_squared(X, means)
            dist2[np.isnan(dist2)] = np.inf
            y = np.argmin(dist2, axis=1)

            # Update means
            for kk in range(self.k):
                means[kk] = X[y==kk].mean(axis=0)
                self.means = means

                #self.means = means
                #print(self.error(X))


            changes = np.sum(y != y_old)
            #print('Running K-means, changes in cluster assignment = {}'.format(changes))

            # Stop if no point changed cluster
            if changes == 0:
                break

        self.means = means

    def predict(self, X):
        means = self.means
        dist2 = euclidean_dist_squared(X, means)
        dist2[np.isnan(dist2)] = np.inf
        return np.argmin(dist2, axis=1)

    def error(self, X):
        # get closest indices from predict()
        indices = self.predict(X)
        
        total = 0
        for i in range(self.means.shape[0]):
            total += np.sum(euclidean_dist_squared(X[indices == i], self.means[[i]]))
        return total

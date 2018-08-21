import numpy as np
import math
from utils import L1_Norm 

class Kmedians:

    def __init__(self, k):
        self.k = k

    def fit(self, X):
        N, D = X.shape
        y = np.ones(N)

        # Getting 500 random samples 
        medians = np.zeros((self.k, D))
        for kk in range(self.k):
            i = np.random.randint(N)
            medians[kk] = X[i]

        self.medians = medians
        #print(self.error(X))

        while True:
            y_old = y

            # Compute L1 norm distance to each median
            dist2 = L1_Norm(X, medians)
            #dist2 = euclidean_dist_squared(X, medians)
            dist2[np.isnan(dist2)] = np.inf
            y = np.argmin(dist2, axis=1)

            # Update medians
            for kk in range(self.k):
                if X[y==kk].size != 0:
                    medians[kk] = np.median(X[y==kk], axis=0)
                    self.medians = medians
                    #print(self.error(X))


            changes = np.sum(y != y_old)
            #print('Running K-means, changes in cluster assignment = {}'.format(changes))

            # Stop if no point changed cluster
            if changes == 0:
                break

        self.medians = medians


    def predict(self, X):
        medians = self.medians
        dist2 = L1_Norm(X, medians)
        #dist2 = euclidean_dist_squared(X, medians)
        dist2[np.isnan(dist2)] = np.inf
        return np.argmin(dist2, axis=1)

    def error(self, X):
        N, D = X.shape
        medians = self.medians
        dist_sum = 0

        # Get all the distances between X values and means
#        dist = euclidean_dist_squared(X, means)
        dist = L1_Norm(X, medians)
        min_index = np.argmin(dist, axis=1)

        # Get sum
        for n in range(N):
            dist_sum = dist_sum + np.amin(dist[n])

        return dist_sum

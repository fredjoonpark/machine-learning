import numpy as np
import utils

from random_tree import RandomTree

class RandomForest:

    # Create a forest of trees
    def __init__(self, max_depth, num_trees):
        self.num_trees = num_trees
        self.max_depth = max_depth

    def fit(self, X, y):
        # Fit each tree
        treeList = []
        for i in range(self.num_trees):
            tree = RandomTree(self.max_depth)
            tree.fit(X,y)
            treeList.append(tree)
        self.treeList = treeList

    def predict(self, X):
        M, D = X.shape
        A = np.zeros((self.num_trees, M))

        # Get prediction from every tree 
        for index, tree in enumerate(self.treeList):
            A[index] = tree.predict(X)

        # Get mode of every column
        y = np.zeros(M)
        for index in range(M):
            y[index]= utils.mode(A[:,index])

        return y

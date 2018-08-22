import numpy as np
from numpy.linalg import solve

# Least Squares with a bias added
class LeastSquaresBias:

    def fit(self,X,y):
        N,D = X.shape
        x = np.ones((N,1))
        X = np.append(x, X, axis=1)
        self.w = solve(X.T@X, X.T@y)

    def predict(self, X):
        N,D = X.shape
        x = np.ones((N,1))
        X = np.append(x, X, axis=1)
        return X@self.w

# Least Squares with polynomial basis
class LeastSquaresPoly:
    def __init__(self, p):
        self.p = p

    def fit(self,X,y):
        Z = self.__polyBasis(X)
        self.w = solve(Z.T@Z, Z.T@y)

    def predict(self, X):
        Z = self.__polyBasis(X)
        return Z@self.w

    # A private helper function to transform any matrix X into
    # the polynomial basis defined by this class at initialization
    # Returns the matrix Z that is the polynomial basis of X.
    def __polyBasis(self, X):
        N,D = X.shape
        Z = np.zeros((N, self.p+1))
        for p in range(self.p+1):
            P = np.power(X,p)
            Z[:,p] = P[:,0]
        return Z

# Least Squares with RBF Kernel
class LeastSquaresRBF:
    def __init__(self, sigma):
        self.sigma = sigma

    def fit(self,X,y):
        self.X = X
        n, d = X.shape

        Z = self.__rbfBasis(X, X, self.sigma)

        # Solve least squares problem
        a = Z.T@Z + 1e-12*np.identity(n) # tiny bit of regularization
        b = Z.T@y
        self.w = solve(a,b)


    def predict(self, Xtest):
        Z = self.__rbfBasis(Xtest, self.X, self.sigma)
        yhat = Z@self.w
        return yhat

    def __rbfBasis(self, X1, X2, sigma):
        n1 = X1.shape[0]
        n2 = X2.shape[0]
        d = X1.shape[1]
        den = 1 / np.sqrt(2 * np.pi * (sigma** 2))

        D = (X1**2)@np.ones((d, n2)) + \
            np.ones((n1, d))@(X2.T** 2) - \
            2 * (X1@X2.T)

        Z = den * np.exp(-1* D / (2 * (sigma**2)))
        return Z

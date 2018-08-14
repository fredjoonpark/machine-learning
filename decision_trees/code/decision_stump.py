import numpy as np
import utils


class DecisionStump:

    def __init__(self):
        pass


    def fit(self, X, y):
        N, D = X.shape

        y_mode = utils.mode(y)

        splitSat = y_mode
        splitVariable = None
        splitValue = None
        splitNot = None

        minError = np.sum(y != y_mode)

        # Check if labels are not all equal
        if np.unique(y).size > 1:
            # Loop over features looking for the best split

            for d in range(D):
                for n in range(N):
                    # Choose value to equate to
                    value = X[n, d]

                    # Find most likely class for each split
                    y_sat = utils.mode(y[X[:,d] > value])
                    y_not = utils.mode(y[X[:,d] <= value])

                    # Make predictions
                    y_pred = y_sat * np.ones(N)
                    y_pred[X[:, d] <= value] = y_not

                    # Compute error
                    errors = np.sum(y_pred != y)

                    # Compare to minimum error so far
                    if errors < minError:
                        # This is the lowest error, store this value
                        minError = errors
                        splitVariable = d
                        splitValue = value
                        splitSat = y_sat
                        splitNot = y_not

        self.splitVariable = splitVariable
        self.splitValue = splitValue
        self.splitSat = splitSat
        self.splitNot = splitNot


    def predict(self, X):

        M, D = X.shape

        if self.splitVariable is None:
            return self.splitSat * np.ones(M)

        yhat = np.zeros(M)

        for m in range(M):
            if X[m, self.splitVariable] > self.splitValue:
                yhat[m] = self.splitSat
            else:
                yhat[m] = self.splitNot

        return yhat

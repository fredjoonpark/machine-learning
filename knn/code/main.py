# standard Python imports
import os
import argparse
import time
import pickle

# 3rd party libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import utils
from knn import KNN

def load_dataset(filename):
    with open(os.path.join('..','data',filename), 'rb') as f:
        return pickle.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--module', required=True,
        choices=["1"])

    io_args = parser.parse_args()
    module = io_args.module

    # K-Nearest Neighbors
    if module == '1':
        dataset = load_dataset("citiesSmall.pkl")
        X, y = dataset["X"], dataset["y"]
        X_test, y_test = dataset["Xtest"], dataset["ytest"]
        #model = KNN(1)
        model = KNN(3)
        #model = KNN(10)

        model.fit(X, y)

        # training error
        y_pred = model.predict(X)
        tr_error = np.mean(y_pred != y)

        # test error
        y_pred = model.predict(X_test)
        te_error = np.mean(y_pred != y_test)

        utils.plotClassifier(model, X, y)
        fname = os.path.join("..", "figs", "knn_classification.pdf")
        plt.savefig(fname)
        pass
        
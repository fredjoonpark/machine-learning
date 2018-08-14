# standard Python imports
import os
import argparse
import time
import pickle

# 3rd party libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeClassifier # see http://scikit-learn.org/stable/install.html

import utils
from decision_stump import DecisionStump
from decision_tree import DecisionTree

def load_dataset(filename):
    with open(os.path.join('..','data',filename), 'rb') as f:
        return pickle.load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--module', required=True,
        choices=["1.1", "1.2", "1.3", "1.4", "1.5"])

    io_args = parser.parse_args()
    module = io_args.module


    # Decision Stump using inequalities/threshold
    if module == "1.1":
        # 1. Load citiesSmall dataset
        dataset = load_dataset("citiesSmall.pkl")
        X = dataset["X"]
        y = dataset["y"]

        # 2. Evaluate decision stump
        model = DecisionStump()
        model.fit(X, y)
        y_pred = model.predict(X)

        error = np.mean(y_pred != y)
        print("Decision Stump with inequality rule error: %.3f" % error)

        # PLOT RESULT
        utils.plotClassifier(model, X, y)

        fname = os.path.join("..", "figs", "decision_stump_boundary.pdf")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)

    # Simple decision tree using decision stumps
    elif module == "1.2":
        # 1. Load citiesSmall dataset
        dataset = load_dataset("citiesSmall.pkl")
        X = dataset["X"]
        y = dataset["y"]

        # 2. Evaluate decision tree
        model = DecisionTree(max_depth=2)
        model.fit(X, y)

        y_pred = model.predict(X)
        error = np.mean(y_pred != y)

        print("Error: %.3f" % error)
    
    # Depth-analysis comparison (my decision tree vs scikit's)
    elif module == "1.3":
        dataset = load_dataset("citiesSmall.pkl")
        X = dataset["X"]
        y = dataset["y"]
        print("n = %d" % X.shape[0])

        depths = np.arange(1,15) # depths to try

        # My decision tree 
        t = time.time()
        my_tree_errors = np.zeros(depths.size)
        for i, max_depth in enumerate(depths):
            model = DecisionTree(max_depth=max_depth)
            model.fit(X, y)
            y_pred = model.predict(X)
            my_tree_errors[i] = np.mean(y_pred != y)
        print("My decision tree took %f seconds" % (time.time()-t))
        
        plt.plot(depths, my_tree_errors, label="mine")
        
        # Scikit's decision tree
        t = time.time()
        sklearn_tree_errors = np.zeros(depths.size)
        for i, max_depth in enumerate(depths):
            model = DecisionTreeClassifier(max_depth=max_depth, criterion='entropy', random_state=1)
            model.fit(X, y)
            y_pred = model.predict(X)
            my_tree_errors[i] = np.mean(y_pred != y)
        print("scikit-learn's decision tree took %f seconds" % (time.time()-t))

        
        plt.plot(depths, my_tree_errors, label="sklearn")
        plt.xlabel("Depth of tree")
        plt.ylabel("Classification error")
        plt.legend()
        fname = os.path.join("..", "figs", "error_comparison.pdf")
        plt.savefig(fname)
        
        tree = DecisionTreeClassifier(max_depth=1)
        tree.fit(X, y)

        y_pred = tree.predict(X)
        error = np.mean(y_pred != y)

        print("Error: %.3f" % error)

    # Plotting error vs depth for scikit's decision tree 
    elif module == "1.4":
        dataset = load_dataset("citiesSmall.pkl")
        X, y = dataset["X"], dataset["y"]
        X_test, y_test = dataset["Xtest"], dataset["ytest"]
        depths = np.arange(1,15) # depths to try
        
        t = time.time()
        tr_error = np.zeros(depths.size)
        te_error = np.zeros(depths.size)
        for i, max_depth in enumerate(depths):
            model = DecisionTreeClassifier(max_depth=max_depth, criterion='entropy', random_state=1)
            model.fit(X, y)
            y_pred = model.predict(X)
            tr_error[i] = np.mean(y_pred != y)
            y_pred = model.predict(X_test)
            te_error[i] = np.mean(y_pred != y_test)
    
        plt.plot(depths, tr_error, label="tr_error")
        plt.plot(depths, te_error, label="te_error")
        plt.xlabel("Depth of tree")
        plt.ylabel("Classification error")
        plt.legend()
        fname = os.path.join("..", "figs", "error_vs_depth.pdf")
        plt.savefig(fname)

        pass

    # Learning through splitting by validation set
    elif module == "1.5":
        dataset = load_dataset("citiesSmall.pkl")
        X, y = dataset["X"], dataset["y"]
        split = round((len(X)+1)/2)
        
        training_set = X[:split] 
        y_training = y[:split]

        validation_set = X[split:] 
        y_validation = y[split:]
        
        depths = np.arange(1,15) # depths to try
        
        tr_error = np.zeros(depths.size)
        te_error = np.zeros(depths.size)
        for i, max_depth in enumerate(depths):
            model = DecisionTreeClassifier(max_depth=max_depth, criterion='entropy', random_state=1)
            model.fit(X, y)

            y_pred = model.predict(training_set)
            tr_error[i] = np.mean(y_pred != y_training)
            y_pred = model.predict(validation_set)
            te_error[i] = np.mean(y_pred != y_validation)
    
        plt.plot(depths, tr_error, label="tr_error")
        plt.plot(depths, te_error, label="te_error")
        plt.xlabel("Depth of tree")
        plt.ylabel("Classification error")
        plt.legend()
        fname = os.path.join("..", "figs", "dt_validation_set.pdf")
        plt.savefig(fname)

        pass

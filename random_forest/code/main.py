# basics
import os
import pickle
import argparse
import matplotlib.pyplot as plt
import numpy as np


# sklearn imports
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from decision_stump import DecisionStumpInfoGain
from decision_tree import DecisionTree
from random_tree import RandomTree
from random_forest import RandomForest 

def plot_2dclustering(X,y):
    plt.scatter(X[:,0], X[:,1], c=y)
    plt.title('Cluster Plot')


def load_dataset(filename):
    with open(os.path.join('..','data',filename), 'rb') as f:
        return pickle.load(f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--module', required=True)

    io_args = parser.parse_args()
    module = io_args.module

    if module == '1':
        dataset = load_dataset('vowel.pkl')
        X = dataset['X']
        y = dataset['y']
        X_test = dataset['Xtest']
        y_test = dataset['ytest']
        print("\nn = %d, d = %d\n" % X.shape)

        def evaluate_model(model):
            model.fit(X,y)

            y_pred = model.predict(X)
            tr_error = np.mean(y_pred != y)

            y_pred = model.predict(X_test)
            te_error = np.mean(y_pred != y_test)
            print("    Training error: %.3f" % tr_error)
            print("    Testing error: %.3f" % te_error)

        print("My implementations:")
        print("  Decision tree info gain")
        evaluate_model(DecisionTree(max_depth=np.inf, stump_class=DecisionStumpInfoGain))
        print("  Random tree info gain")
        evaluate_model(RandomTree(max_depth=np.inf))
        print("  Random forest info gain, 10 trees")
        evaluate_model(RandomForest(max_depth=np.inf, num_trees=10)) # TODO: implement this
        print("  Random forest info gain, 50 trees")
        evaluate_model(RandomForest(max_depth=np.inf, num_trees=50)) # TODO: implement this

        print("sklearn implementations")
        print("  Decision tree info gain")
        evaluate_model(DecisionTreeClassifier(criterion="entropy"))
        print("  Random forest info gain")
        evaluate_model(RandomForestClassifier(criterion="entropy"))
        print("  Random forest info gain, more trees")
        evaluate_model(RandomForestClassifier(criterion="entropy", n_estimators=50))

# basics
import os
import pickle
import argparse
import matplotlib.pyplot as plt
import numpy as np

from kmeans import Kmeans
from kmedians import Kmedians
from sklearn.cluster import DBSCAN



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

    # Visualizing K-means
    if module == '1':
        X = load_dataset('clusterData.pkl')['X']

        model = Kmeans(k=4)
        model.fit(X)
        plot_2dclustering(X, model.predict(X))

        fname = os.path.join("..", "figs", "basic_kmeans_visualization.png")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)

    # Initializaing 50 times to see if kmeans can find best clustering
    elif module == '2':
        X = load_dataset('clusterData.pkl')['X']
        
        best_model = None
        best_error = np.inf
        for i in range(50):
            model = Kmeans(k=4)
            model.fit(X)
            err = model.error(X)
            if err < best_error:
                best_error = err
                best_model = model

        plot_2dclustering(X, best_model.predict(X))

        fname = os.path.join("..", "figs", "50_kmeans_init.png")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)

    # Using the elbow-method to eyeball the best k
    elif module == '3':
        X = load_dataset('clusterData.pkl')['X']

        errors = np.zeros(10)
        for i in range(10):
            min_error = np.inf
            for j in range(50):
                model = Kmeans(k=i+1)
                model.fit(X)
                err = model.error(X)
                if err < min_error:
                    min_error = err
            errors[i] = min_error
        plt.plot(np.arange(1,11), errors)
        plt.title("k Vs min_error")
        plt.xlabel("k")
        plt.ylabel("Min error across 50 random initializations")
        plt.draw()
        fname = os.path.join("..", "figs", "kmeans_elbow_method.png")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)
        #print(errors)

    # Using K-medians for different clustering problems
    elif module == '4':
        X = load_dataset('clusterData2.pkl')['X']

        # using elbow method to eyeball the best k
        errors = np.zeros(10)
        for i in range(10):
            min_error = np.inf
            for j in range(50):
                model = Kmedians(k=i+1)
                model.fit(X)
                err = model.error(X)
                if err < min_error:
                    min_error = err
            errors[i] = min_error
        plt.plot(np.arange(1,11), errors)
        plt.title("k Vs min_error")
        plt.xlabel("k")
        plt.ylabel("Min error across 50 random initializations")
        plt.draw()
        fname = os.path.join("..", "figs", "kmedians_elbow_method.png")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)

    # Trying out scikit's DBSCAN
    elif module == '5':
        X = load_dataset('clusterData2.pkl')['X']
        
        # Play around with hyperparameters
        model = DBSCAN(eps=2, min_samples=3)
        y = model.fit_predict(X)

        print("Labels (-1 is unassigned):", np.unique(model.labels_))

        plot_2dclustering(X,y)
        fname = os.path.join("..", "figs", "dbscan_clustering.png")
        plt.savefig(fname)
        print("\nFigure saved as '%s'" % fname)



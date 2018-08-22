import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

import linear_model

from sklearn.model_selection import train_test_split


def load_dataset(filename):
    with open(os.path.join('..','data',filename), 'rb') as f:
        return pickle.load(f)

def test_and_plot(model,X,y,Xtest=None,ytest=None,title=None,filename=None):

    # Compute training error
    yhat = model.predict(X)
    trainError = np.mean((yhat - y)**2)
    print("Training error = %.1f" % trainError)

    # Compute test error
    if Xtest is not None and ytest is not None:
        yhat = model.predict(Xtest)
        testError = np.mean((yhat - ytest)**2)
        print("Test error     = %.1f" % testError)

    # Plot model
    plt.figure()
    plt.plot(X,y,'b.')
    
    # Choose points to evaluate the function
    Xgrid = np.linspace(np.min(X),np.max(X),1000)[:,None]
    ygrid = model.predict(Xgrid)
    plt.plot(Xgrid, ygrid, 'g')
    
    if title is not None:
        plt.title(title)
    
    if filename is not None:
        filename = os.path.join("..", "figs", filename)
        print("Saving", filename)
        plt.savefig(filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m','--module', required=True)
    io_args = parser.parse_args()
    module = io_args.module

    # Linear regression with a bias variable
    if module == "1":
        data = load_dataset("basisData.pkl")
        X = data['X']
        y = data['y']
        Xtest = data['Xtest']
        ytest = data['ytest']

        model = linear_model.LeastSquaresBias()
        model.fit(X,y)

        test_and_plot(model,X,y,Xtest,ytest,title="Least Squares with bias",filename="least_squares_bias.pdf")

    # Polynomial basis plotting for polynomial order p  
    elif module == "2":
        data = load_dataset("basisData.pkl")
        X = data['X']
        y = data['y']
        Xtest = data['Xtest']
        ytest = data['ytest']

        for p in range(11):
            print("p = %d" % p)
            model = linear_model.LeastSquaresPoly(p)
            model.fit(X,y)
            test_and_plot(model,X,y,Xtest,ytest,title='Least Squares Polynomial p = %d'%p,filename="PolyBasis%d.pdf"%p)
        
        # from the output, we see that it starts to overfit at p = 6!


    # RBF study using cross-validation to find best sigma     
    elif module == "3":
        
        data = load_dataset("basisData.pkl")

        X = data['X']
        y = data['y']
        Xtest = data['Xtest']
        ytest = data['ytest']

        n,d = X.shape

        # Randomize data
        index = np.asarray(range(n))
        np.random.shuffle(index)
        X = X[index]
        y = y[index]

        # Find best value of RBF kernel parameter,
        # training on the train set and validating on the validation set
        minErr = np.inf
        for s in range(-15,16):
            sigma = 2**s

            # 10-fold cross-validation
            for i in range(10):
                begin = int(i * n / 10)
                end = int((i + 1) * n / 10)

                # Split into train/validation sets
                Xvalid = X[begin:end]
                yvalid = y[begin:end]
                Xtrain = np.append(X[:begin,:], X[end:,:], axis=0)
                ytrain = np.append(y[:begin,:], y[end:,:], axis=0)

                # Train on the training set
                model = linear_model.LeastSquaresRBF(sigma)
                model.fit(Xtrain,ytrain)

                # Compute the error on the validation set
                yhat = model.predict(Xvalid)
                validError = np.mean((yhat - yvalid)**2)
                print("Error with sigma = 2^%-3d = %.1f" % (s ,validError))

                # Keep track of the lowest validation error
                if validError < minErr:
                    minErr = validError
                    bestSigma = sigma

        print("Value of sigma that achieved the lowest validation error =", bestSigma)

        # Now fit the model based on the full dataset.
        print("Refitting on full training set...\n")
        model = linear_model.LeastSquaresRBF(bestSigma)
        model.fit(X,y)

        test_and_plot(model,X,y,Xtest,ytest,
            title="Least Squares with RBF kernel and $\sigma={}$".format(bestSigma),
            filename="rbf_using_cross_validation.pdf")

    

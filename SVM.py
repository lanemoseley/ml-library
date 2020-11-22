# Author:      Lane Moseley
# Description: This file demonstrates the usage of the support vector machine
#              module implemented in the ML library.
# Resources Used:
#    Iris Dataset:
#         https://archive.ics.uci.edu/ml/datasets/iris
#         https://gist.github.com/curran/a08a1080b88344b0c8a7

import matplotlib.pyplot as plt
from ML import SupportVectorMachine, plot_svc_decision_function
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import classification_report


def main():
    # IRIS DATASET #############################################################
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

    # Extract the first 100 labels
    y = df.iloc[0:100, 4].values

    # Convert the labels to either 1 or 0
    y = np.where(y == 'Iris-setosa', 0, 1)

    # Extract features from dataset [sepal_length, sepal_width]
    X = df.iloc[0:100, [0, 1]].values

    # plot variables
    title = 'Iris Dataset'
    xlabel = 'Sepal Length [cm]'
    ylabel = 'Sepal Width [cm]'

    # scikit-learn support vector machine
    sk_svm = SVC(kernel='linear', C=1E10)
    sk_svm.fit(X, y)

    # Plot the margins
    plt.title(title + "\nscikit-learn support vector machine")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # Plot the setosa data
    plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
    # Plot the versicolor data
    plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
    # Setup the plot legend
    plt.legend(loc='upper left')

    plot_svc_decision_function(sk_svm)
    plt.show()

    print(title + "\nscikit-learn support vector machine")
    print(classification_report(y, sk_svm.predict(X)))

    # ML.py support vector machine
    ml_svm = SupportVectorMachine()
    ml_svm.fit(X, y)

    # Plot the margins
    plt.title(title + "\nML.py support vector machine")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # Plot the setosa data
    plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
    # Plot the versicolor data
    plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
    # Setup the plot legend
    plt.legend(loc='upper left')

    plot_svc_decision_function(ml_svm)
    plt.show()

    print(title + "\nML.py support vector machine")
    print(classification_report(y, sk_svm.predict(X)))
    ############################################################################

    # IRIS DATASET 2 ###########################################################
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

    # Extract the first 100 labels
    y = df.iloc[0:100, 4].values

    # Convert the labels to either 1 or 0
    y = np.where(y == 'Iris-setosa', 0, 1)

    # Extract features from dataset [petal_width, sepal_length]
    X = df.iloc[0:100, [3, 0]].values

    # plot variables
    title = 'Iris Dataset'
    xlabel = 'Petal Width [cm]'
    ylabel = 'Sepal Length [cm]'

    # scikit-learn support vector machine
    sk_svm = SVC(kernel='linear', C=1E10)
    sk_svm.fit(X, y)

    # Plot the margins
    plt.title(title + "\nscikit-learn support vector machine")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # Plot the setosa data
    plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
    # Plot the versicolor data
    plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
    # Setup the plot legend
    plt.legend(loc='upper left')

    plot_svc_decision_function(sk_svm)
    plt.show()

    print(title + "\nscikit-learn support vector machine")
    print(classification_report(y, sk_svm.predict(X)))

    # ML.py support vector machine
    ml_svm = SupportVectorMachine()
    ml_svm.fit(X, y)

    # Plot the margins
    plt.title(title + "\nML.py support vector machine")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # Plot the setosa data
    plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
    # Plot the versicolor data
    plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
    # Setup the plot legend
    plt.legend(loc='upper left')

    plot_svc_decision_function(ml_svm)
    plt.show()

    print(title + "\nML.py support vector machine")
    print(classification_report(y, sk_svm.predict(X)))
    ####################################################################################################################


if __name__ == "__main__":
    main()

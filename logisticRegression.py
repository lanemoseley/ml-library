# Author:      Lane Moseley
# Description: This file demonstrates the usage of the custom logistic regression
#              module implemented in the ML library.
# Resources Used:
#    Fish Dataset:
#         Included with the project, but also available from Kaggle:
#         https://www.kaggle.com/aungpyaeap/fish-market
#    Iris Dataset:
#         https://archive.ics.uci.edu/ml/datasets/iris
#         https://gist.github.com/curran/a08a1080b88344b0c8a7
#    Logistic Regression Using scikit-learn:
#         https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html

import matplotlib.pyplot as plt
from ML import LogisticRegression, plot_decision_regions
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression as skLogisticRegression
from sklearn.metrics import classification_report


def main():
    # IRIS DATASET #############################################################
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

    # Extract the first 100 labels
    y = df.iloc[0:100, 4].values

    # Convert the labels to either 1 or 0
    y = np.where(y == 'Iris-setosa', 0, 1)

    # Extract features from dataset [sepal_length, petal_length]
    X = df.iloc[0:100, [0, 2]].values

    # plot variables
    title = 'Iris Dataset'
    xlabel = 'Sepal Length [cm]'
    ylabel = 'Petal Length [cm]'

    # Plot what we have so far
    # Plot labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # Plot the setosa data
    plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
    # Plot the versicolor data
    plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')
    # Setup the plot legend
    plt.legend(loc='upper left')

    # Display the plot
    plt.show()

    # scikit-learn logistic regression
    skLR = skLogisticRegression()
    skLR.fit(X, y)

    # plot the decision regions and display metrics to the console
    plot_decision_regions(X, y, skLR, resolution=0.1, x_label=xlabel, y_label=ylabel,
                          title=title + "\nscikit-learn Logistic Regression Model")
    print(title + "\nscikit-learn Logistic Regression Model")
    print(classification_report(y, skLR.predict(X)))

    # ML.py logistic regression
    mlLR = LogisticRegression(learning_rate=0.05, iterations=25)
    mlLR.fit(X, y)

    # plot the decision regions and display metrics to the console
    plot_decision_regions(X, y, mlLR, resolution=0.1, x_label=xlabel, y_label=ylabel,
                          title=title + "\nML.py Logistic Regression Model")
    print(title + "\nML.py Logistic Regression Model")
    print(classification_report(y, mlLR.predict(X)))

    # FISH DATASET #############################################################
    df = pd.read_csv('./Fish.csv')
    df = df.drop(df.index[0:61])        # Parkki rows
    df = df.drop(df.index[11:84])       # Smelt rows

    # Extract the data for Parkki and Smelt fish
    y = df.iloc[:, 0].values

    # Convert the labels to either 0 or 1
    y = np.where(y == 'Parkki', 0, 1)

    # Extract features from dataset [weight, length]
    X = df.iloc[:, [2, 1]].values

    # plot variables
    title = 'Fish Dataset'
    xlabel = 'Length [cm]'
    ylabel = 'Weight [g]'

    # Plot what we have so far
    # Plot labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # Plot the Parkki data
    plt.scatter(X[:11, 0], X[:11, 1], color='red', marker='o', label='Parkki')
    # Plot the Smelt data
    plt.scatter(X[11:, 0], X[11:, 1], color='blue', marker='x', label='Smelt')
    # Setup the plot legend
    plt.legend(loc='upper left')

    # Display the plot
    plt.show()

    # scikit-learn logistic regression
    skLR = skLogisticRegression()
    skLR.fit(X, y)

    # plot the decision regions and display metrics to the console
    plot_decision_regions(X, y, skLR, resolution=0.1, x_label=xlabel, y_label=ylabel,
                          title=title + "\nscikit-learn Logistic Regression Model")
    print(title + "\nscikit-learn Logistic Regression Model")
    print(classification_report(y, skLR.predict(X)))

    # ML.py logistic regression
    mlLR = LogisticRegression(learning_rate=0.01, iterations=10)
    mlLR.fit(X, y)

    # plot the decision regions and display metrics to the console
    plot_decision_regions(X, y, mlLR, resolution=0.1, x_label=xlabel, y_label=ylabel,
                          title=title + "\nML.py Logistic Regression Model")
    print(title + "\nML.py Logistic Regression Model")
    print(classification_report(y, mlLR.predict(X)))


if __name__ == "__main__":
    main()

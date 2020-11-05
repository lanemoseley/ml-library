# Author:      Lane Moseley
# Description: This file demonstrates the usage of the custom Axis Aligned Rectangle
#              classifier implemented in the ML library.
# Resources Used:
#    Fish Dataset:
#         Included with the project, but also available from Kaggle:
#         https://www.kaggle.com/aungpyaeap/fish-market
#    Iris Dataset:
#         https://archive.ics.uci.edu/ml/datasets/iris
#         https://gist.github.com/curran/a08a1080b88344b0c8a7
#    scikit-learn AdaBoostClassifier (for testing low vc learners):
#         https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html
#    Developing scikit-learn Estimators:
#         https://scikit-learn.org/stable/developers/develop.html

import matplotlib.pyplot as plt
from ML import AxisAlignedRectangles, plot_decision_regions
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


def main():
    # IRIS DATASET #####################################################################################################
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

    # Extract the first 100 labels
    Y = df.iloc[0:100, 4].values

    # Convert the labels to either 1 or -1
    Y = np.where(Y == 'Iris-setosa', -1, 1)

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

    # scikit-learn AdaBoost With scikit-learn Pre-built Decision Stump
    boost = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1, max_leaf_nodes=2),
                               algorithm='SAMME', n_estimators=10, learning_rate=1.0)
    boost.fit(X, Y)

    # plot the decision regions and display metrics to the console
    plot_decision_regions(X, Y, boost, 0.01, x_label=xlabel, y_label=ylabel,
                          title=title + "\nscikit-learn Decision Stump Classifier")
    print("scikit-learn AdaBoost With scikit-learn Pre-Built Decision Stump")
    print("Score:", boost.score(X, Y))

    # scikit-learn AdaBoost With Custom Decision Stump
    boost = AdaBoostClassifier(base_estimator=AxisAlignedRectangles(),
                               algorithm='SAMME', n_estimators=10, learning_rate=1.0)
    boost.fit(X, Y)

    # plot the decision regions and display metrics to the console
    plot_decision_regions(X, Y, boost, 0.01, x_label=xlabel, y_label=ylabel,
                          title=title + "\nML.py Axis Aligned Rectangle Classifier")
    print("scikit-learn AdaBoost With ML.py Axis Aligned Rectangle Classifier")
    print("Score:", boost.score(X, Y))
    ####################################################################################################################

    # FISH DATASET 1 ###################################################################################################
    df = pd.read_csv('./Fish.csv')
    df = df.drop(df.index[0:61])        # Parkki rows
    df = df.drop(df.index[11:84])       # Smelt rows

    # Extract the data for Parkki and Smelt fish
    Y = df.iloc[:, 0].values

    # Convert the labels to either 1 or -1
    Y = np.where(Y == 'Parkki', -1, 1)

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

    # scikit-learn AdaBoost With scikit-learn Pre-built Decision Stump
    boost = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1, max_leaf_nodes=2),
                               algorithm='SAMME', n_estimators=10, learning_rate=1.0)
    boost.fit(X, Y)

    # plot the decision regions and display metrics to the console
    plot_decision_regions(X, Y, boost, x_label=xlabel, y_label=ylabel,
                          title=title + "\nscikit-learn Decision Stump Classifier")
    print("\nscikit-learn AdaBoost With scikit-learn Pre-Built Decision Stump")
    print("Score:", boost.score(X, Y))

    # scikit-learn AdaBoost With Custom Decision Stump
    boost = AdaBoostClassifier(base_estimator=AxisAlignedRectangles(),
                               algorithm='SAMME', n_estimators=10, learning_rate=1.0)
    boost.fit(X, Y)

    # plot the decision regions and display metrics to the console
    plot_decision_regions(X, Y, boost, 0.1, x_label=xlabel, y_label=ylabel,
                          title=title + "\nML.py Axis Aligned Rectangle Classifier")
    print("scikit-learn AdaBoost With ML.py Axis Aligned Rectangle Classifier")
    print("Score:", boost.score(X, Y))
    ###################################################################################################################

    # FISH DATASET 2 ###################################################################################################
    df = pd.read_csv('./Fish.csv')

    # Extract the data for Bream and Roach fish
    Y = df.iloc[:55, 0].values

    # Convert the labels to either 1 or -1
    Y = np.where(Y == 'Bream', -1, 1)

    # Extract features from dataset [weight, length]
    X = df.iloc[:55, [2, 1]].values

    # plot variables
    title = 'Fish Dataset'
    xlabel = 'Length [cm]'
    ylabel = 'Weight [g]'

    # Plot what we have so far
    # Plot labels
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # Plot the Bream data
    plt.scatter(X[:35, 0], X[:35, 1], color='red', marker='o', label='Bream')
    # Plot the Roach data
    plt.scatter(X[35:, 0], X[35:, 1], color='blue', marker='x', label='Roach')
    # Setup the plot legend
    plt.legend(loc='upper left')

    # Display the plot
    plt.show()

    # scikit-learn AdaBoost With scikit-learn Pre-built Decision Stump
    boost = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1, max_leaf_nodes=2),
                               algorithm='SAMME', n_estimators=10, learning_rate=1.0)
    boost.fit(X, Y)

    # plot the decision regions and display metrics to the console
    plot_decision_regions(X, Y, boost, resolution=0.1, x_label=xlabel, y_label=ylabel,
                          title=title + "\nscikit-learn Decision Stump Classifier")
    print("\nscikit-learn AdaBoost With scikit-learn Pre-Built Decision Stump")
    print("Score:", boost.score(X, Y))

    # scikit-learn AdaBoost With Custom Decision Stump
    boost = AdaBoostClassifier(base_estimator=AxisAlignedRectangles(),
                               algorithm='SAMME', n_estimators=10, learning_rate=1.0)
    boost.fit(X, Y)

    # plot the decision regions and display metrics to the console
    plot_decision_regions(X, Y, boost, 0.1, x_label=xlabel, y_label=ylabel,
                          title=title + "\nML.py Axis Aligned Rectangle Classifier")
    print("scikit-learn AdaBoost With ML.py Axis Aligned Rectangle Classifier")
    print("Score:", boost.score(X, Y))
    ####################################################################################################################


if __name__ == "__main__":
    main()

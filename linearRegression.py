# Author:      Lane Moseley
# Description: This file demonstrates the usage of the custom Linear Regression
#              module implemented in the ML library.
# Resources Used:
#    Fish Dataset:
#         Included with the project, but also available from Kaggle:
#         https://www.kaggle.com/aungpyaeap/fish-market
#    Iris Dataset:
#         https://archive.ics.uci.edu/ml/datasets/iris
#         https://gist.github.com/curran/a08a1080b88344b0c8a7
#    Performing Linear Regression Using scikit-learn:
#         https://medium.com/analytics-vidhya/linear-regression-using-iris-dataset-hello-world-of-machine-learning-b0feecac9cc1
#         https://scikit-learn.org/stable/auto_examples/linear_model/plot_ols.html

import matplotlib.pyplot as plt
from ML import LinearRegression, plot_regression_line
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression as skLinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split


def demo_helper(X, Y, learning_rate, iterations, title, x_label, y_label, x_lim=None, y_lim=None):
    # Splitting the Dataset
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size= 0.33, random_state= 101)

    # Custom ML linear regression module
    ml_regressor = LinearRegression(learning_rate=learning_rate, iterations=iterations)

    # scikit-learn linear regression module
    scikit_regressor = skLinearRegression()

    # Perform linear regression using both scikit-learn and ML.py library
    ml_regressor.fit(X_train, y_train)      # custom version
    scikit_regressor.fit(X_train, y_train)  # scikit-learn version

    # Make predictions
    mlY_pred = ml_regressor.predict(X_test)
    skY_pred = scikit_regressor.predict(X_test)

    # Plot the X, Y data
    plt.scatter(X, Y)
    plt.title(title + ": Raw Data")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

    # Plot the X, Y_pred data
    custom_label = title + ": ML.py Linear Regression Module\nLearning Rate: " + \
                   str(learning_rate) + ", Iterations: " + str(iterations)
    plot_regression_line(mlY_pred, X_test, y_test, x_label, y_label,
                         custom_label, x_range=x_lim, y_range=y_lim)
    plot_regression_line(skY_pred, X_test, y_test, x_label, y_label,
                         title + ": scikit-learn Linear Regression Module",
                         x_range=x_lim, y_range=y_lim)

    # Evaluate Each Model's Performance
    d_ml = {
        'Metrics': ['Mean Absolute Error:',
                    'Mean Squared Error:',
                    'Mean Root Squared Error:'],
        'Values': [mean_absolute_error(y_test, mlY_pred),
                   mean_squared_error(y_test, mlY_pred),
                   np.sqrt(mean_squared_error(y_test, mlY_pred))]
    }

    d_sk = {
        'Metrics': ['Mean Absolute Error:',
                    'Mean Squared Error:',
                    'Mean Root Squared Error:'],
        'Values': [mean_absolute_error(y_test, skY_pred),
                   mean_squared_error(y_test, skY_pred),
                   np.sqrt(mean_squared_error(y_test, skY_pred))]
    }

    df_ml = pd.DataFrame(data=d_ml)
    df_sk = pd.DataFrame(data=d_sk)

    fig, ax = plt.subplots(2)
    cell_text = []
    for row in range(len(df_ml)):
        cell_text.append(df_ml.iloc[row])

    ax[0].set_title(title + ": ML.py Linear Regression Module\nLearning Rate: " +
                    str(learning_rate) + ", Iterations: " + str(iterations))
    ax[0].table(cellText=cell_text, colLabels=df_ml.columns, loc='center')
    ax[0].axis(False)

    cell_text = []
    for row in range(len(df_sk)):
        cell_text.append(df_sk.iloc[row])

    ax[1].set_title(title + ": scikit-learn Linear Regression Module")
    ax[1].table(cellText=cell_text, colLabels=df_sk.columns, loc='center')
    ax[1].axis(False)
    plt.show()

    print("ML.py Linear Regression Weights for y = mx + b (" + title + "):")
    print("Slope:", ml_regressor.slope)
    print("Intercept:", ml_regressor.intercept, "\n")


def main():
    # IRIS DATASET
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

    # We want to predict Y = Sepal Length using X = Petal Length
    Y = df.iloc[:, 0]
    X = df.iloc[:, 2]
    Y = np.array([Y.to_numpy()]).T
    X = np.array([X.to_numpy()]).T

    # Set some reasonable limits for each graph
    x_label = "Petal Length"
    y_label = "Sepal Length"
    x_lim = (0, 10)
    y_lim = (3, 9)
    learning_rate = 0.05
    iterations = 1000

    demo_helper(X, Y, learning_rate, iterations, "Iris Dataset", x_label,
                y_label, x_lim, y_lim)

    # FISH DATASET
    df = pd.read_csv('./Fish.csv')
    # We want to predict Y = fish weight (g) using X = fish length (cm)
    Y = df.iloc[:, 1]
    X = df.iloc[:, 2]
    Y = np.array([Y.to_numpy()]).T
    X = np.array([X.to_numpy()]).T

    # Set some reasonable limits for each graph
    x_label = "Fish Length (cm)"
    y_label = "Fish Weight (g)"
    # x_lim = (5, 50)
    # y_lim = (0, 1500)
    learning_rate = 0.001
    iterations = 10000

    demo_helper(X, Y, learning_rate, iterations, "Fish Dataset", x_label, y_label)


if __name__ == "__main__":
    main()

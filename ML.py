# Program Name: Machine Learning Library
# Author: Lane Moseley
# Class: CSC 448, Fall 2020
# Professor: Dr. Karlsson
# Language/Compiler: Python 3.7
# Known Bugs:  No known bugs at this time.

import numpy as np
from math import exp, inf
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels


class AxisAlignedRectangles(BaseEstimator, ClassifierMixin):
    """This is the axis-aligned rectangle low vc dimension learner implementation
       for the ML library.

    Args:
        BaseEstimator : Base class for all estimators in scikit-learn,
                        used for compatibility with the sci-kit-learn AdaBoostClassifier
        ClassifierMixin : Mixin class for all classifiers in scikit-learn,
                          used for compatibility with the sci-kit-learn AdaBoostClassifier
    """
    def __init__(self, iterations=10):
        """Initialize the axis-aligned rectangle low vc dimension learner.

        Args:
            iterations: number of iterations for reducing size of axis-aligned rectangle, defaults to 10
        """
        self.__maximums = None
        self.__minimums = None
        self.iterations = iterations

    def fit(self, X, y, sample_weight=None):
        """Fit training data.

        Args:
            X : X training vector
            y : y label vector
            sample_weight (optional): Required for compatibility with the scikit-learn Adaboost module. Defaults to None.

        Returns:
            self : Required for compatibility with the scikit-learn Adaboost module.
        """
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y

        positives = X[y == 1]
        self.__maximums = np.zeros((X.shape[1]))
        self.__minimums = np.zeros((X.shape[1]))

        for col in range(X.shape[1]):
            self.__maximums[col] = max(positives[:, col])
            self.__minimums[col] = min(positives[:, col])

        prediction = self.predict(X)
        error = len(prediction[prediction != y])
        old_max = self.__maximums
        old_min = self.__minimums

        for iter in range(self.iterations):
            prediction = self.predict(X)

            for i in range(len(prediction)):
                if prediction[i] != y[i] and y[i] == -1:
                    dist_to_min = np.linalg.norm(X[i] - self.__minimums)
                    dist_to_max = np.linalg.norm(self.__maximums - X[i])

                    # make the rectangle slightly smaller so that it
                    # excludes the mislabelled point
                    if dist_to_max < dist_to_min:
                        self.__maximums = X[i] - 0.1
                    else:
                        self.__minimums = X[i] - 0.1

                prediction = self.predict(X)
                t_error = len(prediction[prediction != y])

                if t_error < error:
                    error = t_error
                    old_max = self.__maximums
                    old_min = self.__minimums
                else:
                    self.__maximums = old_max
                    self.__minimums = old_min

        return self

    def predict(self, X):
        """Return the predicted Y values.

        Args:
            X : X test vector

        Returns:
            Y_pred : Y prediction vector
        """
        Y_pred = np.ones((X.shape[0], 1))

        for row in range(X.shape[0]):
            for col in range(X.shape[1]):
                if X[row, col] < self.__minimums[col] or X[row, col] > self.__maximums[col]:
                    Y_pred[row] = -1

        return Y_pred.T[0]


class DecisionStump(BaseEstimator, ClassifierMixin):
    """This is the decision stump low vc dimension learner implementation
       for the ML library.

    Args:
        BaseEstimator : Base class for all estimators in scikit-learn,
                        used for compatibility with the sci-kit-learn AdaBoostClassifier
        ClassifierMixin : Mixin class for all classifiers in scikit-learn,
                          used for compatibility with the sci-kit-learn AdaBoostClassifier
    """
    def __init__(self):
        """Initialize the decision stump low vc dimension learner.
        """
        # lambdas for comparisons
        self.__greater = lambda a, b : a > b
        self.__lesser = lambda a, b : a < b

        self.__error = inf
        self.__inequality = self.__greater
        self.__split = 0
        self.__threshold = inf

    def fit(self, X, y, sample_weight=None):
        """Fit training data.

        Args:
            X : X training vector
            y : y label vector
            sample_weight (optional): Required for compatibility with the scikit-learn Adaboost module. Defaults to None.

        Returns:
            self : Required for compatibility with the scikit-learn Adaboost module.
        """
        self.classes_ = unique_labels(y)
        self.X_ = X
        self.y_ = y

        # for each feature in X
        for j in range(X.shape[1]):
            # S = { (x_1, y_1), (x_2, y_2), ..., (x_m, y_m) }
            S = np.hstack((X, np.array([y]).T))

            # sort(S) using jth coordinate s.t. x1j <= x2j <= ... <= xmj
            # where j is the column and 1,2,...m are the rows
            S = S[np.argsort(S[:, j])]

            # remove duplicate values from consideration
            # this can cut down on the number of thresholds that are tested
            keys, indices = np.unique(S[:, j], return_index=True)
            unique = S[indices]

            # for each row in X[:, split], test the midpoint between consecutive
            # feature values to see if it would make a good threshold
            for row in range(0, unique.shape[0] - 1):
                # check the threshold using the greater than lambda
                # save the old configuration
                old_split = self.__split
                old_thresh = self.__threshold
                old_lambda = self.__inequality

                # assign a new configuration
                self.__inequality = self.__greater
                self.__split = j
                self.__threshold = unique[row, j] + ((unique[row+1, j] - unique[row, j]) / 2.0)

                y_pred = self.predict(X)
                error = len(y_pred[y_pred != y])

                if error >= self.__error:
                    self.__split = old_split
                    self.__threshold = old_thresh
                    self.__inequality = old_lambda

                else:
                    self.__error = error

                # check the threshold using the less than lambda
                # save the old configuration
                old_split = self.__split
                old_thresh = self.__threshold
                old_lambda = self.__inequality

                # assign a new configuration
                self.__inequality = self.__lesser
                self.__split = j
                self.__threshold = unique[row, j] + ((unique[row+1, j] - unique[row, j]) / 2.0)

                y_pred = self.predict(X)
                error = len(y_pred[y_pred != y])

                if error >= self.__error:
                    self.__split = old_split
                    self.__threshold = old_thresh
                    self.__inequality = old_lambda

                else:
                    self.__error = error

        return self

    def predict(self, X):
        """Return the predicted Y values.

        Args:
            X : X test vector

        Returns:
            Y_pred : Y prediction vector
        """
        Y_pred = np.ones((X.shape[0], 1))
        Y_pred[self.__inequality(X[:, self.__split], self.__threshold)] = -1

        return Y_pred.T[0]


class LinearRegression:
    """This is the linear regression implementation for the ML library.
    """
    def __init__(self, learning_rate=0.05, iterations=1000):
        """Initialize the linear regression module.

        Args:
            learning_rate (float, optional): used to scale the weight array. Defaults to 0.05.
            iterations (int, optional): number of gradient descent iterations. Defaults to 1000.
        """
        self.intercept = 0
        self.__iterations = iterations
        self.__learning_rate = learning_rate
        self.slope = 0

    def fit(self, X, Y):
        """Fit training data. Since all linear regression problems are
        convex regardless of the input data, we can use gradient descent
        without worrying about getting stuck in a local minimum.

        Args:
            X : X training vector (independent variables)
            Y : Y training vector (dependent variables)
        """
        self.intercept = 0
        self.slope = 0
        n = float(len(X))

        # equation for a line:  Y = mX + b
        # mean squared error:   1/n * sum(0, n, (Y[i] - Y_pred[i])^2)
        #                     = 1/n * sum(0, n, (Y[i] - (mX[i] + b)^2)

        # naive batch gradient descent using partial derivatives
        # of the mean squared error function (see doc for more details)
        for i in range(self.__iterations):
            # predicted y value with current weights
            pred = (self.slope * X) + self.intercept

            # derivative of mean squared error w.r.t. m (slope)
            d_slope_avg = sum(2 * (Y - pred) * (-X)) / n

            # derivative of mean squared error w.r.t. b (intercept)
            d_intercept_avg = sum(-2 * (Y - pred)) / n

            # update the weights
            self.slope -= (self.__learning_rate * d_slope_avg)
            self.intercept -= (self.__learning_rate * d_intercept_avg)

    def predict(self, X_test):
        """Return the predicted Y values.

        Args:
            X_test : X test vector

        Returns:
            Y_pred : Y prediction vector
        """
        return (self.slope * X_test) + self.intercept


class LogisticRegression:
    """This is the logistic regression implementation for the ML library.
    """
    def __init__(self, learning_rate=0.01, iterations=10):
        """Initialize the logistic regression module.

        Args:
            learning_rate (float, optional): used to scale the weight array. Defaults to 0.01.
            iterations (int, optional): number of gradient descent iterations. Defaults to 10.
        """
        self.__iterations = iterations
        self.__learning_rate = learning_rate
        self.__weights = None

    def fit(self, X, Y):
        """Fit training data.

        Args:
            X: X training vector (independent variables)
            Y : Y training vector (dependent variables)
        """
        self.__weights = np.zeros(X.shape[1] + 1)

        for iter in range(self.__iterations):
            for i in range(X.shape[0]):
                y_pred = 1.0 / (1.0 + exp(-(self.__weights[0] + np.dot(X[i], self.__weights[1:]))))
                error = Y[i] - y_pred
                self.__weights[0] += self.__learning_rate * error * y_pred * (1.0 - y_pred)
                self.__weights[1:] += self.__learning_rate * error * y_pred * (1.0 - y_pred) * X[i]

    def predict(self, X_test):
        """Return the predicted Y values.

        Args:
            X_test: X_test : X test vector

        Returns:
            Y_pred : Y prediction vector
        """
        # apply the weights
        X_test = np.dot(X_test, self.__weights[1:])
        X_test += self.__weights[0]

        # apply the sigmoid function
        Y_pred = 1.0 / (1.0 + np.exp(-X_test))

        # apply a threshold to the result for binary prediction
        Y_pred[Y_pred >= 0.5] = 1
        Y_pred[Y_pred < 0.5] = 0

        # return the array as type int
        return Y_pred.astype(int)


class Perceptron:
    """This is the perceptron implementation for the ML library.
    """
    def __init__(self, learning_rate=0.01, iterations=10):
        """Initialize the perceptron.

        Args:
            learning_rate Float: used to scale the weight array
            iterations Int: number of iterations for fitting data to labels
        """
        self.__errors = []
        self.__iterations = iterations
        self.__learning_rate = learning_rate
        self.__weight = np.array([])

    @property
    def errors(self):
        """This is the getter for the error array. Using a getter prevents the caller from changing the array.

        Returns:
            list: the array of errors in each iteration
        """
        return self.__errors

    def fit(self, X, y):
        """Fit training data.

        Args:
            X : Training vectors, X.shape : [#samples, #features]
            y : Target values, y.shape : [#samples]
        """
        # create a weight array of size X.size + 1 (weight[0] is the bias)
        # initialize all elements to zero
        self.__weight = np.zeros(X[0].size + 1)

        # Number of mis-classifications, creates an array
        # to hold the number of mis-classifications
        self.__errors = []

        # main loop to fit the data to the labels
        for i in range(self.__iterations):
            # set iteration error to zero
            self.__errors.append(0)

            # loop over all the objects in X and corresponding y element
            for xi, target in zip(X, y):
                prediction = self.predict(xi)

                # calculate the needed (delta_w) update
                delta_w = self.__learning_rate * (target - prediction)

                # update the bias using the current delta_w
                self.__weight[0] += delta_w

                # calculate what the current object will add to the weight and update the weight
                for j in range(len(xi)):
                    self.__weight[j + 1] += delta_w * xi[j]

                # increase the iteration error if delta_w != 0
                if delta_w != 0:
                    self.__errors[i] += 1

            # stop early if the perceptron converges before the provided number of iterations
            if self.__errors[i] == 0:
                return self

        return self

    def net_input(self, X):
        """Calculate the net input.

        Args:
            X : Training vectors, X.shape : [#samples, #features]

        Returns:
            Float: the dot product (X.w) plus the bias
        """
        # Return the dot product: X.w + bias
        return np.dot(X, self.__weight[1:]) + self.__weight[0]

    def predict(self, X):
        """Return the class label after unit step

        Args:
            X : Training vectors, X.shape : [#samples, #features]

        Returns:
            Int: the predicted class label (1 or -1)
        """
        return np.where(self.net_input(X) >= 0.0, 1, -1)

    @property
    def weight(self):
        """This is the getter for the weight array. Using a getter prevents the caller from changing the array.

        Returns:
            numpy.ndarray: the current weight array
        """
        return self.__weight


def plot_decision_regions(X, y, classifier, resolution=0.02, x_label="", y_label="", title=""):
    """This is a helper function to plot the decision regions of the classifier. This shows the partition(s) between the
    different classes of objects.

    Args:
        X : Training vectors, X.shape : [#samples, #features]
        y : Target values, y.shape : [#samples]
        classifier : the classification algorithm
        resolution (float, optional) : the resolution of the meshgrid
        x_label (string, optional) : the x label for the plot
        y_label (string, optional) : the y label for the plot
        title (string, optional) : the title for the plot
    """
    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - .1, X[:, 0].max() + .1
    x2_min, x2_max = X[:, 1].min() - .1, X[:, 1].max() + .1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], alpha=0.8, color=cmap(idx),
                    marker=markers[idx], label=cl)

    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend(loc='upper left')
    plt.show()


def plot_regression_line(y_predicted, x_actual, y_actual, x_label="", y_label="",
                         title="", line_color='red', x_range=None, y_range=None):
    """ This function plots the linear regression line.
        A linear regression line has an equation of the form Y = a + bX.
        X is the explanatory variable and Y is the dependent variable.
        The slope of the line is b, and a is the intercept.

    Args:
        y_predicted : the y values predicted by the linear regression learner
        x_actual : the actual x values
        y_actual : the actual y values
        x_label (str, optional): Horizontal axis label. Defaults to "".
        y_label (str, optional): Vertical axis label. Defaults to "".
        title (str, optional): Plot title. Defaults to "".
        line_color (str, optional): Plot line color. Defaults to 'red'.
        x_range (tuple, optional): x range of graph
        y_range (tuple, optional): y range of graph
    """

    plt.scatter(x_actual, y_actual)
    plt.plot(x_actual, y_predicted, color=line_color)

    if x_range is not None and y_range is not None:
        plt.xlim(x_range)
        plt.ylim(y_range)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

    plt.show()

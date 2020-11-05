# Author: Lane Moseley
# Description: This file demonstrates the usage of the custom Perceptron
#              module implemented in the ML library.

import numpy as np
import matplotlib.pyplot as plt
from ML import Perceptron, plot_decision_regions
import pandas as pd


df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

# Extract the first 100 labels
y = df.iloc[0:100, 4].values
# Convert the labels to either 1 or -1
y = np.where(y == 'Iris-setosa', -1, 1)

# Extract features from dataset [sepal_length, petal_length]
X = df.iloc[0:100, [0, 2]].values

# Plot what we have so far
# Plot labels
plt.title('Iris Dataset')
plt.xlabel('Sepal Length [cm]')
plt.ylabel('Petal Length [cm]')

# Plot the setosa data
plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')

# Plot the versicolor data
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')

# Setup the plot legend
plt.legend(loc='upper left')

# Display the plot
plt.show()

# Setup the Perceptron
pn = Perceptron(0.1, 10)

# Fit X to y (i.e. find the weights)
pn.fit(X, y)

# Print the error array
print("Errors:", pn.errors)

# Plot the results of the first fit
plt.plot(range(1, len(pn.errors) + 1), pn.errors, marker='o')
plt.title('Iris Dataset')
plt.xlabel('Iteration')
plt.ylabel('# of Misclassifications')
plt.show()

print("Net Input X:", pn.net_input(X))
print("Predict X:", pn.predict(X))
print("Weights:", pn.weight)

plot_decision_regions(X, y, pn, x_label='Sepal Length [cm]', y_label='Petal Length [cm]', title='Iris Dataset')

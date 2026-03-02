import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

# Load dataset
iris = load_iris()

# Use only 2 classes (Setosa vs Versicolor)
X = iris.data[:, [0]]  # sepal length
y = iris.target

# Make it binary
y = (y == 0).astype(int)

# Train model
model = LogisticRegression()
model.fit(X, y)

# Create values for decision boundary
X_test = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)
y_prob = model.predict_proba(X_test)[:, 1]

# Plot
plt.scatter(X, y)
plt.plot(X_test, y_prob, color='red')
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Probability of Setosa")
plt.title("Logistic Regression on Iris Dataset")
plt.show()
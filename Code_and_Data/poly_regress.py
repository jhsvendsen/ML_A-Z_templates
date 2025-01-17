# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Importing the dataset
os.chdir("C:/Users/Johan PC/Desktop/Machine Learning A-Z/Part 2 - Regression/Section 6 - Polynomial Regression/Polynomial_Regression")
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values # 1:2 to make sure it is considered as matrix
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
""" from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting linear poly regression
# Linear
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y) 

# Polynomial fitting
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
linreg2 = LinearRegression()
linreg2.fit(X_poly,y)

# Plotting Linear Regression Model
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear)')
plt.xlabel('Pos level')
plt.ylabel('Salary')
plt.show

# Plotting Poly Linear model
X_grid = np.arange(min(X),max(X),0.01)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, linreg2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Poly)')
plt.xlabel('Pos level')
plt.ylabel('Salary')
plt.show

# Prediction of Salary
lin_reg.predict(6.5)
# Classification
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

os.chdir("C:/Users/Johan PC/Desktop/Machine Learning A-Z/Part 3 - Classification/Section 14 - Logistic Regression/Logistic_Regression")

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2,3]].values
y = dataset.iloc[:, 4].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
"""sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting the Regression Model to the dataset
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting a new result
y_pred = classifier.predict(X_test)

# Evaluating Results
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
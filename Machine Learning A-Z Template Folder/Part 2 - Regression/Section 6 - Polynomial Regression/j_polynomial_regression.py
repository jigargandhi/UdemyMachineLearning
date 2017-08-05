# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

# creating a linear model for comparison
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X_poly = poly_reg.fit_transform(X)

lin_reg2 = LinearRegression()
lin_reg.fit(X_poly,y)

#visualize linear result
plt.scatter(X,y,color='red')
plt.plot(X, np.array([x[0] for x in lin_reg.predict(X.reshape(-1,1))]), color='blue')
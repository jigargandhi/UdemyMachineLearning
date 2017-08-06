# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as pt

dataset= pd.read_csv('data.csv')
X = dataset.iloc[:,-1].values
y = dataset.iloc[:3].values

"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 0)"""


# Fitting the Regression Model to the dataset
# Create your regressor here

# Predicting a new result
y_pred = regressor.predict(6.5)


plt.scatter(X,y, color ='red')


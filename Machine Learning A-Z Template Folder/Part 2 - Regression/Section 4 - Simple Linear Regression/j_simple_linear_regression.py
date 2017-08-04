# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,0].values

Y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)

#no need to do feature scaling as library is going to take care of it

#creating machine learning model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
#reshape after v0.19
regressor.fit(X_train.reshape(-1,1), y_train.reshape(-1,1))


#predicting the test results
y_pred = regressor.predict(X_test.reshape(-1,1))


#visualizing the training set
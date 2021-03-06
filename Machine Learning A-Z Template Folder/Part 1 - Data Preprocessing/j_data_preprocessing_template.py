# -*- coding: utf-8 -*-
"""
Created on Thu Aug  3 15:51:12 2017

@author: gdi
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing data
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,3].values

# Take care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy ='mean',axis=0)
imputer = imputer.fit(X[:,1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# convert label to numbers
labelEncoder_X = LabelEncoder()
X[:,0]= labelEncoder_X.fit_transform(X[:,0])
#encode labels as different values in columns
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()
labelEncoder_Y = LabelEncoder()
y= labelEncoder_Y.fit_transform(Y)

#split training set and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X =StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
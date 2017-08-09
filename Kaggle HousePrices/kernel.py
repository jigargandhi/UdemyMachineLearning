# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train= pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
#replacing numerical categorical values
train['MSSubClass'].replace({20:'C20',30:'C30', 40:'C40', 45: 'C45', 50: 'C50'
                           , 60: 'C60', 70: 'C70', 75: 'C75', 80: 'C80', 85: 'C85', 90:  'C90' 
                           , 120: 'C120', 150:'C150',160:'C160', 180:'C180', 
                             190: 'C190'}, inplace=True)
test['MSSubClass'].replace({20:'C20',30:'C30', 40:'C40', 45: 'C45', 50: 'C50'
                           , 60: 'C60', 70: 'C70', 75: 'C75', 80: 'C80', 85: 'C85', 90:  'C90' 
                           , 120: 'C120', 150:'C150',160:'C160', 180:'C180', 
                             190: 'C190'}, inplace= True)	
#imputing
for col in test.columns:
    if (str(test[col].dtype) == 'object'):
        
        train[col].fillna('',inplace=True)
        test[col].fillna('',inplace=True)

for col in test.columns:
    if(str(test[col].dtype== np.dtype('float64'))):
        train[col].fillna(0,inplace=True)
        if(col != 'SalePrice'):
                test[col].fillna(0,inplace=True)


from sklearn.preprocessing import LabelEncoder
#encoding labels
for col in test.columns:
    le = LabelEncoder()    
    if(str(train[col].dtype) =='object'):
        
        data = train[col].append(test[col])
        le.fit(data.values)
        train[col] = le.transform(train[col])
        test[col] = le.transform(test[col])

#standard scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
train['SalePrice'] = scaler.fit_transform(train['SalePrice'])
        
X = train.iloc[:,1:-1].values
y = train.iloc[:,-1].values
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.2, random_state = 0 )        
        
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
# RMSE: 3282784944.2

y_pred2 = regressor.predict(test.iloc[:,1:].values)

from sklearn.metrics import mean_squared_error
rmse = mean_squared_error(y_test, y_pred)

# Create a data frame out of the y_pred and upload

#visualization ???
plt.scatter(test['GrLivArea'], y_pred2, color='red')
plt.show()
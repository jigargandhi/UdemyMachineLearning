# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset= pd.read_csv('data/train.csv')
X_test = pd.read_csv('data/test.csv')
#removing the Id, SalePrice column as they do not make sense in the X
X_train = dataset.iloc[:,:-1]
# y is the sale price
y = dataset.iloc[:,80].values

X = X_train.append(X_test,  ignore_index=True)
X_withoutIndex = pd.get_dummies(X, drop_first= True)


X = X_withoutIndex.iloc[:,1:].values

X_train = X


##Label Encoding is required, OneHotEncoder is required, Scaling is not required
##, Imputing is required for Alley column
#from sklearn.preprocessing import LabelEncoder, OneHotEncoder
##label_mssubClass = LabelEncoder()
#X[:,0]= label_mssubClass.fit_transform(X[:,0])
#label_mszoning = LabelEncoder()
#X[:,1] = label_mszoning
#hotencoder = OneHotEncoder()
#X= hotencoder.fit_transform(X)

#transform Non Linear models use decision tree, random forest, and 
#multiple linear Regression


from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

import statsmodels.formula.api as sm
regressor_ols = sm.OLS(endog = y_train, exog = X_train).fit()
regressor_ols.summary()
"""                    OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.945
Model:                            OLS   Adj. R-squared:                  0.931
Method:                 Least Squares   F-statistic:                     67.26
Date:                Mon, 07 Aug 2017   Prob (F-statistic):               0.00
Time:                        14:50:59   Log-Likelihood:                -13132.
No. Observations:                1168   AIC:                         2.674e+04
Df Residuals:                     931   BIC:                         2.794e+04
Df Model:                         236                                         
Covariance Type:            nonrobust                                         
==============================================================================
"""

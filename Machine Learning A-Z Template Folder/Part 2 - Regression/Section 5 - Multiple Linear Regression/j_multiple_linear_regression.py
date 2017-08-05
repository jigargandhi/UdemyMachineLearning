# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[: ,: -1].values
y = dataset.iloc[:,4].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# convert label to numbers
labelEncoder_X = LabelEncoder()
X[:,3]= labelEncoder_X.fit_transform(X[:,3])
#encode labels as different values in columns
#in array use index of column
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()

#avoiding the dummy variable

X = X[: ,1:]

#splitting training and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state=0)



# fitting multiple 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#predict the regressor
y_pred = regressor.predict(X_test)

# backward elimination
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regressor_ols = sm.OLS(endog = y, exog = X_opt).fit()
regressor_ols.summary()
X_opt = X[:, [0, 3, 4, 5]]
regressor_ols = sm.OLS(endog = y, exog = X_opt).fit()
regressor_ols.summary()
X_opt = X[:, [0, 3, 5]]
regressor_ols = sm.OLS(endog = y, exog = X_opt).fit()
regressor_ols.summary()
X_opt = X[:, [0, 3]]
regressor_ols = sm.OLS(endog = y, exog = X_opt).fit()
regressor_ols.summary()





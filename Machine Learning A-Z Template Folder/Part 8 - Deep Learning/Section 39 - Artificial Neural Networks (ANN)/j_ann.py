# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, -1].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X_1 = LabelEncoder()
X[:,1] = labelEncoder_X_1.fit_transform(X[:,1])
labelEncoder_X_2 = LabelEncoder()
X[:,2] = labelEncoder_X_2.fit_transform(X[:,2])
onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()

X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# ANN does require Feature Scaling compulsorily
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Fitting classifier to the Training set
# Part 2 ANN
# Importing libraries
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising ANN

classifier = Sequential()
#1st layer
classifier.add(Dense(output_dim = 6, init ='uniform', activation= 'relu', input_dim= 11 ))
# 2nd layer
classifier.add(Dense(output_dim = 6, init ='uniform', activation= 'relu' ))
#3rd layer
classifier.add(Dense(output_dim = 1, init ='uniform', activation= 'sigmoid'))
# if more than 2 categories activation fun = softmax output_dim = no of categories 

#compile the classifier
# loss 2= = binary _crossentropy, >2 categorical_crossentropy
classifier.compile(optimizer = 'adam', loss =  'binary_crossentropy', metrics =['accuracy'])

classifier.fit(X_train, y_train, batch_size= 10, nb_epoch = 100)

y_pred = classifier.predict(X_test)

# Part 3 Evaluating the model


y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix
cm  = confusion_matrix(y_test, y_pred)



# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


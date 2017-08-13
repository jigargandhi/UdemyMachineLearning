
#Titanic

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

train = pd.read_csv("input/train.csv")
test = pd.read_csv("input/test.csv")
X = train.iloc[:,[2,4,5]].values
Y = train.iloc[:,[1]].values
Z = test.iloc[:,[1,3,4]].values
""" 
Variables of interest:
    pClass, Sex, Age, 

Treatment:
    pClass: Ordinal Variable no treatment required
    Sex: LabelEncoder, OneHotEncoder not required as values are only male and female
    Age, Impute with mean
"""
# Label Encoding Sex
from sklearn.preprocessing import LabelEncoder
labelEncoder = LabelEncoder()
X[:,1] = labelEncoder.fit_transform(X[:,1])
Z[:,1]= labelEncoder.transform(Z[:,1])
# Imputing Age
from sklearn.preprocessing import Imputer
imputer = Imputer()
imputer.fit(X[:,[2]])
X[:,[2]]= imputer.transform(X[:,[2]])
Z[:,[2]] = imputer.transform(Z[:,[2]])

"""
Models:
    LogisticRegression, SVM, KernelSVM, Naive Bayes, Decision Tree, Random Forest
"""
def printAccuracy(d):
    return (d[0,0]+d[1,1])/(d[0,0]+d[1,1]+d[0,1]+d[1,0])

#Logistics Regression
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LogisticRegression
linear_classifier = LogisticRegression()
linear_classifier .fit(X_train, y_train)
y_pred = linear_classifier .predict(X_test)


from sklearn.metrics import confusion_matrix
linear_cm= confusion_matrix(y_test, y_pred)

# [[93,17],[21,48]]

#SVM
from sklearn.svm import SVC
svc_classifier = SVC(kernel = 'rbf', random_state = 0)
svc_classifier.fit(X_train, y_train)
y_pred= svc_classifier.predict(X_test)
svc_cm = confusion_matrix(y_test, y_pred)


#81%

#RandomForest
from sklearn.ensemble import RandomForestClassifier
rf_classifier = RandomForestClassifier(criterion='entropy', random_state=0)
rf_classifier.fit(X_train, y_train)
# Predicting the Test set results
y_pred = rf_classifier.predict(X_test)

rf_cm = confusion_matrix(y_test,y_pred)

# Naive Bayes
from sklearn.naive_bayes import GaussianNB
gnb_classifier = GaussianNB()
gnb_classifier.fit(X_train, y_train)
y_pred = gnb_classifier.predict(X_test)
gnb_cm = confusion_matrix(y_test, y_pred)
# 82%
"""
Evaluation:
    a. Confusion Matrix, CAP Curve (how?)
"""

print("Linear: ",printAccuracy(linear_cm))
print("SVM: ",printAccuracy(svc_cm))
print("Random Forest: ",printAccuracy(rf_cm))
print("Naive Bayes:", printAccuracy(gnb_cm))



"""
Writing Random Forest result to Output
"""
y_result = rf_classifier.predict(Z)
result = pd.DataFrame({"PassengerId":test.iloc[:,0].values, "Survived": y_result})
result.to_csv("output/result.csv",index = None)

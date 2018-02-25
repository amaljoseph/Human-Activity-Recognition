# Importing libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Importing the dataset and preprocessing
train_data = pd.read_csv('train.csv')
X = train_data.iloc[:, :-1].values
y = train_data.iloc[:, 562].values

test_data = pd.read_csv('test.csv')
X_test = test_data.iloc[:, :-1].values
y_test = test_data.iloc[:, 562].values

#Encoding categrical data
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y)
y_test = labelencoder.fit_transform(y_test)

#Fitting the SVM to dataset
from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(X, y)

#Predicting the Test Set results
y_pred = classifier.predict(X_test)


#Making the confusin matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
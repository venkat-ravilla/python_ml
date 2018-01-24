# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 16:06:36 2017

@author: venkataramana.r
"""

import numpy as np
from sklearn import preprocessing, cross_validation, neighbors
import pandas as pd

# read the data set
df = pd.read_csv('breast-cancer-wisconsin.data.txt')
# replace all missing values(?) with -99999
df.replace('?', -99999, inplace=True)
# drop id column since it is not a useful feature 
df.drop(['id'], 1, inplace = True)

#input data
X = np.array(df.drop(['class'],1))
# output data
y = np.array(df['class'])

# Split input data to training and test data
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)

# initialize k means classifier
clf = neighbors.KNeighborsClassifier()

# train the classifier
clf.fit(X_train,y_train)

# find the accuracy
accuracy = clf.score(X_test,y_test)
print(accuracy)

# unknown sample for prediction
example_measures = np.array([[4,2,1,1,1,2,3,2,1],[4,2,1,1,1,2,3,2,1]])
# avoid deprecation errors, len defines the number of samples
example_measures = example_measures.reshape(len(example_measures),-1)

# prediction
prediction = clf.predict(example_measures)
print(prediction)
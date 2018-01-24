# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 15:39:39 2017

@author: venkataramana.r
"""

import pandas as pd
import quandl, math, datetime, time
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use('ggplot')
df = quandl.get("WIKI/GOOGL")
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
df['HL_PCT']=(df['Adj. High'] - df['Adj. Close'])/df['Adj. Close'] * 100.0
df['PCT_change']=(df['Adj. Close'] - df['Adj. Open'])/df['Adj. Open'] * 100.0
df = df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna(-99999,inplace=True)
# assigning label with shifting the forecast_col rows upwards with the integer that is defined below
forecast_out = int(math.ceil(0.01*len(df)))
df['label'] = df[forecast_col].shift(-forecast_out)

# creating input
X = np.array(df.drop(['label'],1))

# scaling the input values between 0 and 1
X = preprocessing.scale(X)

# taking the set of rows to be predicted or which doesnt have any forecast in input set
X_lately = X[-forecast_out:]
# removing the rows which doesnt have any labels or na labels
X = X[:-forecast_out]
df.dropna(inplace=True)
y = np.array(df['label'])

# Split input data to training and test data
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.2)

# Initialize a classifier , n_jobs is for assigning number of threads, -1 indicate as many threads that your system can accomodate
clf = LinearRegression(n_jobs=-1)

# Fit the classifier with the training data
clf.fit(X_train, y_train)

# store a trained model into a file
with open('linearregression.pickle','wb') as f:
    pickle.dump(clf, f)

# read the model from a file
pickle_in = open('linearregression.pickle','rb')
# load the model from a file
clf = pickle.load(pickle_in)

# finding the confidence/accuracy for the classifier
accuracy = clf.score(X_test,y_test)

# forecast/predict the unknown values
forecast_set = clf.predict(X_lately)

print(forecast_set, accuracy, forecast_out)

# defining forcast values
df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = time.mktime(last_date.timetuple())
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix+= one_day
    df.loc[next_date]=[np.nan for _ in range(len(df.columns)-1)] + [i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

# Initialize SVM classifier , provide kernel name and compare the accuracy for different kernels
clf = svm.SVR(kernel='poly')

# Fit the classifier with the training data
clf.fit(X_train, y_train)

# finding the confidence/accuracy of the classifier
accuracy = clf.score(X_test,y_test)

print(accuracy)
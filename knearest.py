# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 16:53:15 2017

@author: venkataramana.r
"""
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
import warnings
from matplotlib import style
from collections import Counter
import pandas as pd
import random

style.use('fivethirtyeight')

# sample dataset with 2 classes k and r
dataset = {'k':[[1,2],[2,3],[3,1]], 'r':[[6,5],[7,7],[8,6]]}
# new input for prediction
new_features = [5,7]

# function for k nearest neighbors algorithm, data:input data, predict:input for prediction , k: number of nearest neighbors
def k_nearest_neighbors(data, predict, k=3):
    # len(data) is the number of classes, k should always be more than number of classes
    if len(data) >=k:
        warnings.warn('K is set to a value less than total voting groups!')
    distances = []
    # creating a list of lists with Euclidean distance between datapoint, predict and datapoint's class EX: [[2.0, 'r'], [2.2360679774997898, 'r'], [3.1622776601683795, 'r']] 
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([euclidean_distance, group])
    
    # sort the list based on distances and take top k sublists and return a the class of of each sublist EX:['r', 'r', 'r']
    votes = [i[1] for i in sorted(distances)[:k]]
    # count the votes and create a list of tuples with class and vote count, take the tuple with most votes and return the class name of that tuple
    vote_result = Counter(votes).most_common(1)[0][0]
    confidence = Counter(votes).most_common(1)[0][1] / k
    return vote_result, confidence
    
result, confidence = k_nearest_neighbors(dataset, new_features, k=3)
print(result)

# read the data set
df = pd.read_csv('breast-cancer-wisconsin.data.txt')
# replace all missing values(?) with -99999
df.replace('?', -99999, inplace=True)
# drop id column since it is not a useful feature 
df.drop(['id'], 1, inplace = True)
full_data = df.astype(float).values.tolist()
random.shuffle(full_data)

test_size = 0.2
train_set = {2:[], 4:[]}
test_set = {2:[], 4:[]}

train_data = full_data[:-int(test_size*len(full_data))]
test_data = full_data[-int(test_size*len(full_data)):]

for i in train_data:
    train_set[i[-1]].append(i[:-1])
    

for i in test_data:
    test_set[i[-1]].append(i[:-1])

correct = 0
total = 0

for group in test_set:
    for data in test_set[group]:
        vote,confidence = k_nearest_neighbors(train_set, data, k=5)
        if group == vote:
            correct +=1
        else:
            print(confidence)
        total +=1

print('Accuracy:',float(correct)/total)
# plot the dataset
[[plt.scatter(ii[0],ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]
plt.scatter(new_features[0], new_features[1], s=100,color = result)
plt.show()
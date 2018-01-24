# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 15:45:48 2017

@author: venkataramana.r
"""

from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random 

style.use('fivethirtyeight')

# sample inputs & outputs
xs = np.array([1,2,3,4,5,6], dtype = np.float64)
ys = np.array([5,4,6,5,6,7], dtype = np.float64)

# random generation of input and output
def create_dataset(hm, variance, step = 2, correlation = False):
    val = 1
    ys = []
    for i in range(hm):
        y = val + random.randrange(-variance,variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val+=step
        elif correlation and correlation == 'neg':
            val-=step
    xs = [i for i in range(len(ys))]
    return np.array(xs, dtype = np.float64), np.array(ys, dtype = np.float64)

def best_fit_slope_and_intercept(xs,ys):
    mx = mean(xs)
    my = mean(ys)
    m = (((mx*my) - mean(xs*ys))/((mx*mx)-mean(xs*xs)))
    b= mean(ys) - m*mean(xs)
    return m, b
    
# function for finding squared error
def squared_error(ys_orig, ys_line):
    return sum((ys_line-ys_orig)**2)
    
# function to find r squared (r2 = 1-(SE_yreg / SE_ymean))
def coefficient_of_determination(ys_orig, ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    squared_error_regr = squared_error(ys_orig, ys_line)
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)
    return 1 - (squared_error_regr / squared_error_y_mean)

xs, ys = create_dataset(40, 40, 2, correlation='pos')

# slope and intercept of the line in linear regression
m,b = best_fit_slope_and_intercept(xs,ys)

# y values for regression line
regression_line = [(m*x)+b for x in xs]
predict_x = 8
predict_y = (m*predict_x)+b

# r-squared value
r_squared = coefficient_of_determination(ys, regression_line)
print(r_squared)

plt.scatter(xs,ys)
plt.scatter(predict_x,predict_y, s=100, color = 'g')
plt.plot(xs,regression_line)
plt.show()
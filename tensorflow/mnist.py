# -*- coding: utf-8 -*-
"""
Created on Wed Jan 03 16:25:55 2018

@author: venkataramana.r
"""
import tensorflow as tf
# importing the data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

#initializing the input data place holder , None indicates that the dimention can be of any length
x = tf.placeholder(tf.float32, [None, 784])

# defining the weights and biases and initializing them with zeros
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# actual output
y_ = tf.placeholder(tf.float32, [None, 10])

# softmax output
y = tf.nn.softmax(tf.matmul(x, W) + b)

# loss function, use tf.nn.softmax_cross_entropy_with_logits instead
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# training the model to reduce loss, here 0.5 is learning rate
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# launching themodel in an interactive session
sess = tf.InteractiveSession()

# Initializing the variables
tf.global_variables_initializer().run()

# running the training step 1000 times with different batches of input
for _ in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
  
# evaluating the model, here tf.argmax gives the index of the highest element in th list which is the label
# The output is a list of boolean
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

# we cast the boolean to float and find the mean which is the accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# We find the accuracy
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

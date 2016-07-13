'''
Tensorflow implementation of some Autoencoders (AE) as a scikit-learn like model 
with fit, transform methods.

@author: Zichen Wang (wangzc921@gmail.com)

@references:

https://github.com/tensorflow/models/tree/master/autoencoder/autoencoder_models
'''

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np
import tensorflow as tf
import tensorflow.contrib import learn

class DualObjectiveAutoencoder(object):
	"""docstring for DualObjectiveAutoencoder"""
	def __init__(self, n_input, hidden_units, 
		dropout_probability=1.0,
		n_classes=2,
		objective='cross_entropy',
		activation_function=tf.nn.softplus, 
		optimizer=tf.train.AdamOptimizer()):
		"""Initializes a DualObjectiveAutoencoder instance.
		Args:
			n_input: Number of input features
			hidden_units: a list of ints specifying the hidden units in each layers
			objective: 'cross_entropy' for classification and 'mse' for regression
		"""
		self.n_input = n_input
		self.hidden_units = hidden_units
		self.activate = activation_function
		self.objective = objective
		self.n_classes = n_classes
		self.dropout_probability = dropout_probability

		network_weights = self._initialize_weights()
		self.weights = network_weights

		# Model
		self.x = tf.placeholder(tf.float32, [None, self.n_input])
		self.y = tf.placeholder(tf.float32, [None, self.n_classes])
		self.keep_prob = tf.placeholder(tf.float32)

		# Encoding
		for i, n_hidden in enumerate(self.hidden_units):
			W = self.weights['encoder%d_W' % i]
			b = self.weights['encoder%d_b' % i]
			if i == 0:
				tensor_in = tf.nn.dropout(self.x, keep_prob=self.keep_prob)
			else:
				tensor_in = hidden
			hidden = self.activate(tf.matmul(tensor_in, W) + b)

		self.z = hidden
		# Decoding
		hidden_units_rev = self.hidden_units[::-1]
		for i, n_hidden in enumerate(hidden_units_rev):
			W = self.weights['decoder%d_W' % i]
			b = self.weights['decoder%d_b' % i]
			if i == 0:
				tensor_in = self.z
			else:
				tensor_in = hidden
			hidden = self.activate(tf.matmul(tensor_in, W) + b)
		self.reconstruction = hidden

		# Loss
		self.reconstruction_loss = tf.reduce_mean(
			tf.square(tf.sub(self.reconstruction, self.x)))
		if self.objective == 'cross_entropy':
			self.supervised_loss = tf.reduce_mean(
				tf.nn.softmax_cross_entropy_with_logits(self.z, self.y))
		elif self.objective == 'mse':
			self.supervised_loss = tf.reduce_mean(
				tf.square(tf.sub(self.z, self.y)))

		self.loss = self.reconstruction_loss + self.supervised_loss
		self.optimizer = optimizer.minimize(self.loss)

		init_op = tf.initialize_all_variables()
		self.sess = tf.Session()
		self.sess.run(init_op)


	def _initialize_weights(self):
		all_weights = dict()
		# Encoding layers
		for i, n_hidden in enumerate(self.hidden_units):
			weight_name = 'encoder%d_W' % i
			bias_name = 'encoder%d_b' % i
			if i == 0:
				weight_shape = [self.n_input, n_hidden]
			else:
				weight_shape = [self.hidden_units[i-1], n_hidden]

			all_weights[weight_name] = tf.get_variable(weight_name, weight_shape, 
				initializer=tf.contrib.layers.xavier_initializer())
			all_weights[bias_name] = tf.get_variable(bias_name, [n_hidden],
				initializer=tf.constant_initializer(0.0))
		
		# Decoding layers
		hidden_units_rev = self.hidden_units[::-1]
		for i, n_hidden in enumerate(hidden_units_rev):
			weight_name = 'decoder%d_W' % i
			bias_name = 'decoder%d_b' % i
			if i != len(hidden_units_rev) - 1: # not the last layer
				weight_shape = [n_hidden, hidden_units_rev[i+1]]
			else:
				weight_shape = [n_hidden, self.n_input]

			all_weights[weight_name] = tf.get_variable(weight_name, weight_shape, 
				initializer=tf.contrib.layers.xavier_initializer())
			all_weights[bias_name] = tf.get_variable(bias_name, [n_hidden],
				initializer=tf.constant_initializer(0.0))

		return all_weights


	def partial_fit(self, X, y):
		loss, opt = self.sess.run((self.loss, self.optimizer), 
			feed_dict={self.x: X, self.y: y, self.keep_prob: self.dropout_probability})
		return loss

	def calc_total_cost(self, X, y):
		return self.sess.run(self.loss, 
			feed_dict={self.x: X, self.y: y, self.keep_prob: 1.0})

	def transform(self, X):
		return self.sess.run(self.z, feed_dict={self.x: X, self.keep_prob: 1.0})

	def predict(self, X):
		return self.sess.run(self.z, 
			feed_dict={self.x: X, self.keep_prob: self.dropout_probability})

	def reconstruct(self, X):
		return self.sess.run(self.reconstruction, feed_dict={self.x: X, self.keep_prob: 1.0})



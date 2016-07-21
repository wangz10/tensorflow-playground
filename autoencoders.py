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

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow.contrib import learn
from sklearn.base import BaseEstimator


class BaseAutoencoder(BaseEstimator):
	"""Base class for autoencoders"""
	def __init__(self, n_input, n_hidden, activation_func='softplus', 
		optimizer_name='AdamOptimizer',
		learning_rate=0.001,
		logdir='/tmp',
		log_every_n=100, 
		seed=42):
		'''
		params:
		
		activation_func (string): a name of activation_func in tf.nn 
		optimizer_name (string): a name of the optimizer object name tf.train
		'''
		self.n_input = n_input
		self.n_hidden = n_hidden
		self.activation_func = activation_func
		self.optimizer_name = optimizer_name
		self.learning_rate = learning_rate
		self.logdir = logdir
		self.log_every_n = log_every_n
		self.seed = seed

		self._init_all()


	def _init_all(self):
		'''Initializes everything after params are bound'''
		self.global_step = 0

		# Count how many encoding layers are there
		if isinstance(self.n_hidden, list):
			self.n_layers = len(self.n_hidden)
			self.hidden_layer_sizes = self.n_hidden
		else:
			self.n_layers = 1
			self.hidden_layer_sizes = [self.n_hidden]

		self._parse_args()

		self._init_graph()

		self.sess = tf.Session(graph=self.graph)
		self.sess.run(self.init_op)

		self.summary_writer = tf.train.SummaryWriter(self.logdir, self.sess.graph)


	def _parse_args(self):
		'''Parse json serializable args to bind objects and functions'''
		self.transfer = eval('tf.nn.%s' % self.activation_func)
		self.optimizer = eval('tf.train.%s(%f)' % (self.optimizer_name, self.learning_rate))


	def _init_graph(self):
		self.graph = tf.Graph()
		with self.graph.as_default():
			tf.set_random_seed(self.seed)

			self.variables = self._init_variables()
			
			# Model
			self.x = tf.placeholder(tf.float32, [None, self.n_input],
				name='x')
			# Encoding
			for i in range(self.n_layers):
				if i == 0:
					tensor_in = self.x
				else:
					tensor_in = hidden

				hidden = self.transfer(tf.add(
					tf.matmul(tensor_in, self.variables['encoder%d_W' % i]), self.variables['encoder%d_b' % i]),
					name='encode_hidden%d' % i)
			# Code
			self.hidden = hidden
			# Decoding
			for i in range(self.n_layers):
				if i == 0:
					tensor_in = self.hidden
				else:
					tensor_in = hidden
				hidden = tf.add(
					tf.matmul(tensor_in, self.variables['decoder%d_W' % i]), self.variables['decoder%d_b' % i],
					name='decode_hidden%d' % i)
			self.z = hidden

			# Reconstruction loss
			# self.loss = tf.mul(0.5, tf.reduce_sum(tf.square(tf.sub(self.z, self.x))), 
			self.loss = tf.reduce_mean(tf.square(tf.sub(self.z, self.x)),	
				name='Reconstruction_loss')
			tf.scalar_summary(self.loss.op.name, self.loss)

			self.optimize_op = self.optimizer.minimize(self.loss)

			self.init_op = tf.initialize_all_variables()
			# To save model
			self.saver = tf.train.Saver()
			# Summary writer for tensorboard
			self.summary_op = tf.merge_all_summaries()


	def _init_variables(self):
		variables = dict()
		# Encoding layers
		for i, n_hidden in enumerate(self.hidden_layer_sizes):
			layer_name = 'encoder%d' % i
			W_name = '%s_W' % layer_name
			b_name = '%s_b' % layer_name
			if i == 0:
				weight_shape = [self.n_input, n_hidden]
			else:
				weight_shape = [self.hidden_layer_sizes[i-1], n_hidden]

			variables[W_name] = tf.get_variable(W_name, weight_shape, 
				initializer=tf.contrib.layers.xavier_initializer())
			variables[b_name] = tf.get_variable(b_name, [n_hidden],
				initializer=tf.constant_initializer(0.0))
		# Decoding layers
		rev_hidden_layer_sizes = self.hidden_layer_sizes[::-1]
		for i, n_hidden in enumerate(rev_hidden_layer_sizes):
			layer_name = 'decoder%d' % i
			W_name = '%s_W' % layer_name
			b_name = '%s_b' % layer_name
			if i == self.n_layers - 1: # the output layer
				weight_shape = [n_hidden, self.n_input]
				b_shape = [self.n_input]
			else:
				weight_shape = [n_hidden, rev_hidden_layer_sizes[i+1]]
				b_shape = rev_hidden_layer_sizes[i+1]

			variables[W_name] = tf.get_variable(W_name, weight_shape, 
				initializer=tf.contrib.layers.xavier_initializer())
			variables[b_name] = tf.get_variable(b_name, b_shape,
				initializer=tf.constant_initializer(0.0))

		return variables 

	def _to_write_summary(self):
		'''Decide whether to write summary at current step'''
		to_write_summary = False
		if self.log_every_n is not None:
			if self.global_step % self.log_every_n == 0:
				to_write_summary = True
		return to_write_summary

	def partial_fit(self, X):
		if self._to_write_summary():
			loss, opt, summary_str = self.sess.run((self.loss, self.optimize_op, self.summary_op), 
				feed_dict={self.x: X})
			self._write_summary(summary_str)
		else: # do not write summary
			loss, opt = self.sess.run((self.loss, self.optimize_op), 
				feed_dict={self.x: X})
		self.global_step += 1
		return loss

	def _write_summary(self, summary_str):
		# Update the events file.
		self.summary_writer.add_summary(summary_str, self.global_step)
		self.summary_writer.flush()

	# def fit(self, X, batch_size=128):
	# 	return 

	def calc_total_cost(self, X):
		return self.sess.run(self.loss, feed_dict = {self.x: X})

	def transform(self, X):
		return self.sess.run(self.hidden, feed_dict={self.x: X})

	def generate(self, hidden=None):
		if hidden is None:
			hidden = np.random.normal(size=self.weights["encoder_b"])
		return self.sess.run(self.z, feed_dict={self.hidden: hidden})

	def reconstruct(self, X):
		return self.sess.run(self.z, feed_dict={self.x: X})

	def get_variable(self, var_name):
		"""Get a variable by name"""
		return self.sess.run(self.variables[var_name])

	def save(self, path):
		'''
		To save trained model and its params.
		'''
		# Create if dir does not exists
		if not os.path.isdir(path):
			os.mkdir(path)

		save_path = self.saver.save(self.sess, 
			os.path.join(path, 'model.ckpt'),
			global_step=self.global_step)
		# save parameters of the model
		params = self.get_params()
		json.dump(params, 
			open(os.path.join(path, 'model_params.json'), 'wb'))

		print("Model saved in file: %s" % save_path)
		return save_path

	def _restore(self, path):
		with self.graph.as_default():
			self.saver.restore(self.sess, path)
	
	@classmethod
	def restore(cls, path):
		'''
		To restore a saved model.
		'''
		# load params of the model
		path_dir = os.path.dirname(path)
		params = json.load(open(os.path.join(path_dir, 'model_params.json'), 'rb'))
		# init an instance of this class
		estimator = cls(**params)
		estimator._restore(path)
		# bind global_step
		global_step = int(path.split('-')[-1])
		estimator.global_step = global_step
		return estimator


class AdditiveGaussianNoiseAutoencoder(BaseAutoencoder):
	
	def __init__(self, n_input, n_hidden, 
		activation_func='softplus', 
		optimizer_name='AdamOptimizer',
		learning_rate=0.001,
		logdir='/tmp',
		log_every_n=100, 
		seed=42,
		noise_stddev=0.01, 
		):
		'''
		params in addition to BaseAutoencoder:

		noise_stddev (float): standard deviation of the Gaussian noise.
		'''
		# Additional args
		self.noise_stddev = noise_stddev

		super(AdditiveGaussianNoiseAutoencoder, self).__init__(
			n_input, n_hidden, 
			activation_func=activation_func, 
			optimizer_name=optimizer_name,
			learning_rate=learning_rate,
			logdir=logdir,
			log_every_n=log_every_n, 
			seed=seed
			)


	def _init_graph(self):
		self.graph = tf.Graph()
		with self.graph.as_default():
			tf.set_random_seed(self.seed)

			self.variables = self._init_variables()

			# Model
			self.x = tf.placeholder(tf.float32, [None, self.n_input],
				name='x')
			# Gaussian noise to be added to the input
			noise = tf.random_normal((self.n_input, ), stddev=self.noise_stddev, 
				name='Gaussian_noise')
			# Encoding
			for i in range(self.n_layers):
				if i == 0:
					tensor_in = self.x + noise
				else:
					tensor_in = hidden

				hidden = self.transfer(tf.add(
					tf.matmul(tensor_in, self.variables['encoder%d_W' % i]), self.variables['encoder%d_b' % i]),
					name='encode_hidden%d' % i)
			# Code
			self.hidden = hidden
			# Decoding
			for i in range(self.n_layers):
				if i == 0:
					tensor_in = self.hidden
				else:
					tensor_in = hidden
				hidden = tf.add(
					tf.matmul(tensor_in, self.variables['decoder%d_W' % i]), self.variables['decoder%d_b' % i],
					name='decode_hidden%d' % i)
			self.z = hidden

			# Reconstruction loss
			# self.loss = tf.mul(0.5, tf.reduce_sum(tf.square(tf.sub(self.z, self.x))), 
			self.loss = tf.reduce_mean(tf.square(tf.sub(self.z, self.x)),
				name='Reconstruction_loss')
			tf.scalar_summary(self.loss.op.name, self.loss)

			self.optimize_op = self.optimizer.minimize(self.loss)

			self.init_op = tf.initialize_all_variables()
			# To save model
			self.saver = tf.train.Saver()
			# Summary writer for tensorboard
			self.summary_op = tf.merge_all_summaries()


class MaskingNoiseAutoencoder(BaseAutoencoder):
	"""docstring for MaskingNoiseAutoencoder"""
	def __init__(self, n_input, n_hidden, 
		activation_func='softplus', 
		optimizer_name='AdamOptimizer',
		learning_rate=0.001,
		logdir='/tmp',
		log_every_n=100, 
		seed=42,
		dropout_probability=0.95
		):
		'''
		params in addition to BaseAutoencoder:

		noise_stddev (float): standard deviation of the Gaussian noise.
		'''
		# Additional args
		self.dropout_probability = dropout_probability

		super(MaskingNoiseAutoencoder, self).__init__(
			n_input, n_hidden, 
			activation_func=activation_func, 
			optimizer_name=optimizer_name,
			learning_rate=learning_rate,
			logdir=logdir,
			log_every_n=log_every_n, 
			seed=seed,
			)

	
	def _init_graph(self):
		self.graph = tf.Graph()
		with self.graph.as_default():
			tf.set_random_seed(self.seed)

			self.variables = self._init_variables()

			self.keep_prob = tf.placeholder((tf.float32), name='keep_prob')
			
			# Model
			self.x = tf.placeholder(tf.float32, [None, self.n_input], name='x')
			# Encoding
			for i in range(self.n_layers):
				if i == 0:
					# Apply dropout on x during training phase only
					tensor_in = tf.nn.dropout(self.x, self.keep_prob)
				else:
					tensor_in = hidden

				hidden = self.transfer(tf.add(
					tf.matmul(tensor_in, self.variables['encoder%d_W' % i]), self.variables['encoder%d_b' % i]),
					name='encode_hidden%d' % i)
			# Code
			self.hidden = hidden
			# Decoding
			for i in range(self.n_layers):
				if i == 0:
					tensor_in = self.hidden
				else:
					tensor_in = hidden
				hidden = tf.add(
					tf.matmul(tensor_in, self.variables['decoder%d_W' % i]), self.variables['decoder%d_b' % i],
					name='decode_hidden%d' % i)
			self.z = hidden

			# Reconstruction loss
			# self.loss = tf.mul(0.5, tf.reduce_sum(tf.square(tf.sub(self.z, self.x))), 
			self.loss = tf.reduce_mean(tf.square(tf.sub(self.z, self.x)),
				name='Reconstruction_loss')
			tf.scalar_summary(self.loss.op.name, self.loss)

			self.optimize_op = self.optimizer.minimize(self.loss)

			self.init_op = tf.initialize_all_variables()
			# To save model
			self.saver = tf.train.Saver()
			# Summary writer for tensorboard
			self.summary_op = tf.merge_all_summaries()

	def partial_fit(self, X):
		if self._to_write_summary():
			loss, opt, summary_str = self.sess.run(
				(self.loss, self.optimize_op, self.summary_op), 
				feed_dict={self.x: X, self.keep_prob: self.dropout_probability})
			self._write_summary(summary_str)
		else:
			loss, opt = self.sess.run(
				(self.loss, self.optimize_op), 
				feed_dict={self.x: X, self.keep_prob: self.dropout_probability})
		self.global_step += 1
		return loss

	def calc_total_cost(self, X):
		return self.sess.run(self.loss, 
			feed_dict={self.x: X, self.keep_prob: 1.0})

	def transform(self, X):
		return self.sess.run(self.hidden, 
			feed_dict={self.x: X, self.keep_prob: 1.0})

	def reconstruct(self, X):
		return self.sess.run(self.z, 
			feed_dict={self.x: X, self.keep_prob: 1.0})

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



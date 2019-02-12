'''
Tensorflow implementation of Neural Variational Document Model(NVDM) algorithm as a scikit-learn like model 
with fit, transform methods.

@author: Zichen Wang (wangzc921@gmail.com)

@references:
https://arxiv.org/abs/1511.06038
https://github.com/ysmiao/nvdm/blob/master/nvdm.py
'''

import collections
import math
import os
import random
import re
import json

import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin

# Set random seeds
SEED = 2019
random.seed(SEED)
np.random.seed(SEED)

# Utils 
def parse_activation_function(function_name):
    '''Given activation funtion name (e.g. tanh, sigmoid, ...), 
    returns the function.
    '''
    return eval('tf.nn.%s' % function_name)

def variable_parser(var_list, prefix):
    '''Return a subset of the all_variables by prefix.'''
    ret_list = []
    for var in var_list:
        varname = var.name
        varprefix = varname.split('/')[0]
        if varprefix == prefix:
            ret_list.append(var)
    return ret_list

class NVDM(BaseEstimator, TransformerMixin):
    def __init__(self, 
        vocab_size=2000,
        n_hidden=500,
        n_topic=50,
        n_sample=1,
        non_linearity='tanh',
        learning_rate=5e-5,
        batch_size=128
        ):
        self.vocab_size = vocab_size
        self.n_hidden = n_hidden
        self.n_topic = n_topic
        self.n_sample = n_sample
        self.non_linearity = non_linearity
        self.learning_rate = learning_rate
        self.batch_size = batch_size
    
        # init all variables in a tensorflow graph
        self._init_graph()

        # create a session
        self.sess = tf.Session(graph=self.graph)
        self.sess.run(self.init_op)

    def _init_graph(self):
        '''
        Init a tensorflow Graph containing:
        input data, variables, model, loss function, optimizer
        '''
        self.graph = tf.Graph()
        with self.graph.as_default():

            # Input data.
            self.x = tf.placeholder(tf.float32, [None, self.vocab_size], name='input')
            # Model.
            ## encoder
            with tf.variable_scope('encoder'): 
                self.enc_vec = tf.contrib.layers.fully_connected(inputs=self.x,
                    num_outputs=self.n_hidden,
                    activation_fn=parse_activation_function(self.non_linearity),
                )
                self.mean = tf.contrib.layers.fully_connected(inputs=self.enc_vec,
                    num_outputs=self.n_topic,
                    activation_fn=None,
                    scope='mean'    
                )
                self.logsigm = tf.contrib.layers.fully_connected(inputs=self.enc_vec,
                    num_outputs=self.n_topic,
                    activation_fn=None,
                    weights_initializer=tf.zeros_initializer(),
                    biases_initializer=tf.zeros_initializer(),
                    scope='logsigm'
                )
                # KL-divergence
                self.kld = -0.5 * tf.reduce_sum(1 - tf.square(self.mean) + 2 * self.logsigm - tf.exp(2 * self.logsigm), 1)
                
            ## decoder
            with tf.variable_scope('decoder'):
                if self.n_sample == 1:  # single sample
                    eps = tf.random_normal(tf.stack([tf.shape(self.x)[0], self.n_topic]), 0, 1)
                    doc_vec = tf.multiply(tf.exp(self.logsigm), eps) + self.mean
                    logits = tf.nn.log_softmax(
                        tf.contrib.layers.fully_connected(inputs=doc_vec, 
                            num_outputs=self.vocab_size,
                            activation_fn=None,
                            scope='projection'
                        )
                    )
                    self.recons_loss = -tf.reduce_sum(tf.multiply(logits, self.x), 1)
                
                else: # multiple samples
                    eps = tf.random_normal(tf.stack([self.n_sample*tf.shape(self.x)[0], self.n_topic]), 0, 1)
                    eps_list = tf.split(0, self.n_sample, eps)
                    recons_loss_list = []
                    for i in xrange(self.n_sample):
                        if i > 0: tf.get_variable_scope().reuse_variables()
                        curr_eps = eps_list[i]
                        doc_vec = tf.multiply(tf.exp(self.logsigm), curr_eps) + self.mean
                        logits = tf.nn.log_softmax(
                            tf.contrib.layers.fully_connected(inputs=doc_vec, 
                                num_outputs=self.vocab_size,
                                activation_fn=None,
                                scope='projection'
                            )
                        )
                        recons_loss_list.append(-tf.reduce_sum(tf.multiply(logits, self.x), 1))
                    self.recons_loss = tf.add_n(recons_loss_list) / self.n_sample

            self.loss = self.recons_loss + self.kld

            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            fullvars = tf.trainable_variables()

            enc_vars = variable_parser(fullvars, 'encoder')
            dec_vars = variable_parser(fullvars, 'decoder')

            enc_grads = tf.gradients(self.loss, enc_vars)
            dec_grads = tf.gradients(self.loss, dec_vars)

            self.optim_enc = optimizer.apply_gradients(zip(enc_grads, enc_vars))
            self.optim_dec = optimizer.apply_gradients(zip(dec_grads, dec_vars))

            # init op 
            self.init_op = tf.global_variables_initializer()
            # create a saver 
            self.saver = tf.train.Saver()

    def fit(self, X, n_epochs=1, alternate_epochs=10, shuffle=True):
        '''Learn model from data X 
        @params:
            - X: sparse matrix of term-frequencies, shape=(n_documents, vocab_size)
            - n_epochs: number of epochs to train
            - alternate_epochs: number of alternates between updating encoder/decoder variables
            - shuffle: whether to shuffle the samples before each epoch
        '''
        batch_size = self.batch_size
        n_train_docs = X.shape[0]
        train_data_idx = np.arange(n_train_docs)
        for epoch in range(n_epochs):
            if shuffle: 
                np.random.shuffle(train_data_idx)
            for switch in (0, 1):
                # switching between updating encoder and decoder
                if switch == 0:
                    optim = self.optim_dec
                    print_mode = 'updating decoder'
                else:
                    optim = self.optim_enc
                    print_mode = 'updating encoder'

                for i in range(alternate_epochs):

                    loss_sum = 0.0
                    kld_sum = 0.0
                    ppx_sum = 0.0
                    word_count = 0
                    doc_count = 0

                    n_batches = math.ceil(n_train_docs / batch_size)

                    for idx_batch in range(n_batches):
                        start_idx = idx_batch * batch_size
                        end_idx = (idx_batch+1) * batch_size
                        data_batch = X[train_data_idx][start_idx:end_idx].toarray()
                        
                        count_batch = np.squeeze(np.asarray(data_batch.sum(axis=1)))

                        _, (loss, kld) = self.sess.run((optim, 
                                                [self.loss, self.kld]),
                                                feed_dict={self.x: data_batch})

                        loss_sum += np.sum(loss)
                        kld_sum += np.sum(kld) / data_batch.shape[0]
                        word_count += np.sum(count_batch)
                        # to avoid nan error
                        count_batch = np.add(count_batch, 1e-12)
                    
                    # perplexity
                    ppx = np.exp(loss_sum / word_count)
                    kld = kld_sum/n_batches
                    print('| Epoch train: {:d} |'.format(epoch+1), 
                        print_mode, '{:d}'.format(i),
                        '| Corpus ppx: {:.5f}'.format(ppx),  # perplexity for all docs
                        '| KLD: {:.5}'.format(kld))

        return self

    def transform(self, X):
        return self.sess.run(self.mean, feed_dict={self.x: X.toarray()})

    def perplexity(self, X):
        losses = self.sess.run(self.loss, feed_dict={self.x: X.toarray()})
        word_count = X.sum()
        ppx = np.exp(np.sum(losses) / word_count)
        return ppx


    def save(self, path):
        '''To save trained model and its params.
        '''
        save_path = self.saver.save(self.sess, 
            os.path.join(path, 'model.ckpt'))
        # save parameters of the model
        params = {
            'vocab_size': self.vocab_size,
            'n_hidden': self.n_hidden,
            'n_topic': self.n_topic,
            'n_sample': self.n_sample,
            'non_linearity': self.non_linearity,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size
        }
        json.dump(params, 
            open(os.path.join(path, 'model_params.json'), 'w'))
        return save_path

    def _restore(self, path):
        with self.graph.as_default():
            self.saver.restore(self.sess, path)

    @classmethod
    def load(cls, path):
        '''To restore a saved model.
        '''
        # load params of the model
        params = json.load(open(os.path.join(path, 'model_params.json'), 'r'))
        # init an instance of this class
        estimator = cls(**params)
        estimator._restore(os.path.join(path, 'model.ckpt'))
        return estimator


'''
Non-modulized Implementation of paragram vector.

@ref: http://arxiv.org/abs/1405.4053
'''
import math
import random
import collections
from itertools import compress

import numpy as np
import pandas as pd
import tensorflow as tf
from nltk import word_tokenize
from nltk.tokenize import RegexpTokenizer

from word2vec import build_dataset

tokenizer0 = RegexpTokenizer(r"(?u)\b\w\w+\b")

SEED = 2016
vocabulary_size = 50000

def read_treebank(nrows=100):
	'''
	Read sentences from treebank datasetSentences.txt file and tokenize them.

	Return a list of token lists, each of which is from the same sentence.
	'''
	df = pd.read_csv('data/stanfordSentimentTreebank/datasetSentences.txt',
		sep='\t',
		nrows=nrows)
	# return df['sentence'].map(word_tokenize).tolist()
	return df['sentence'].map(tokenizer0.tokenize).tolist()

docs = read_treebank()
# print docs[:4]
print len(docs[0])

def concat_lists(a, b):
	'''Concat two lists
	'''
	# a.extend(b)
	return a + b

## 
def build_doc_dataset(docs, vocabulary_size=50000):
	'''
	Build the dictionary and replace rare words with UNK token.
	
	Parameters
	----------
	docs: list of token lists, each token list represent a sentence
	vocabulary_size: maximum number of top occurring tokens to produce, 
		rare tokens will be replaced by 'UNK'
	'''
	count = [['UNK', -1]]
	words = reduce(concat_lists, docs)

	doc_ids = [] # collect document(sentence) indices
	for i, doc in enumerate(docs):
		doc_ids.extend([i] * len(doc))

	word_ids, count, dictionary, reverse_dictionary = build_dataset(words, vocabulary_size=vocabulary_size)

	return doc_ids, word_ids, count, dictionary, reverse_dictionary

doc_ids, word_ids, count, dictionary, reverse_dictionary = build_doc_dataset(docs, vocabulary_size=vocabulary_size)


data_index = 0

def generate_batch_pvdm(batch_size, window_size):
	'''
	Batch generator for PV-DM (Distributed Memory Model of Paragraph Vectors).
	batch should be a shape of (batch_size, window_size+1)

	Parameters
	----------
	batch_size: number of words in each mini-batch
	window_size: number of leading words on before the target word direction 
	'''
	global data_index
	assert batch_size % window_size == 0
	batch = np.ndarray(shape=(batch_size, window_size + 1), dtype=np.int32)
	labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
	span = window_size + 1
	buffer = collections.deque(maxlen=span) # used for collecting word_ids[data_index] in the sliding window
	buffer_doc = collections.deque(maxlen=span) # collecting id of documents in the sliding window
	# collect the first window of words
	for _ in range(span):
		buffer.append(word_ids[data_index])
		buffer_doc.append(doc_ids[data_index])
		data_index = (data_index + 1) % len(word_ids)

	mask = [1] * span
	mask[-1] = 0 
	i = 0
	while i < batch_size:
		if len(set(buffer_doc)) == 1:
			doc_id = buffer_doc[-1]
			# all leading words and the doc_id
			batch[i, :] = list(compress(buffer, mask)) + [doc_id]
			labels[i, 0] = buffer[-1] # the last word at end of the sliding window
			i += 1
			# print buffer
			# print list(compress(buffer, mask))
		# move the sliding window  
		buffer.append(word_ids[data_index])
		buffer_doc.append(doc_ids[data_index])
		data_index = (data_index + 1) % len(word_ids)

	return batch, labels

## examinng the batch generator function
batch_size = 50
window_size = 10

batch, labels = generate_batch_pvdm(batch_size, window_size)

print batch
print labels

print word_ids[:10]
print doc_ids[:10]

print 'The sentences:'
print docs[0]
print docs[1]

print 'batch:'
for row in batch:
	print [reverse_dictionary[i] for i in row[:-1]] + [row[-1]]
print 'labels:'
print [reverse_dictionary[i] for i in labels.reshape(batch_size)]



## The computation graph
batch_size = 128
window_size = 8

embedding_size_w = 64
embedding_size_d = 64
document_size = len(docs)
n_neg_samples = 32
optimize = 'Adagrad'
learning_rate = 1.0
# print document_size

## choose some sentences for validation
valid_size = 5
valid_examples = np.array(random.sample(range(100), 5))

graph = tf.Graph()
with graph.as_default():
	# Set graph level random seed
	tf.set_random_seed(SEED)
	# Input data.
	train_dataset = tf.placeholder(tf.int32, shape=[batch_size, window_size+1])
	train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
	valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

	# Variables.
	# embeddings for words, W in paper
	word_embeddings = tf.Variable(
		tf.random_uniform([vocabulary_size, embedding_size_w], -1.0, 1.0))

	# embedding for documents (can be sentences or paragraph), D in paper
	doc_embeddings = tf.Variable(
		tf.random_uniform([document_size, embedding_size_d], -1.0, 1.0))

	concatenated_embed_vector_length = embedding_size_w * window_size + embedding_size_d

	# softmax weights, W and D vectors should be concatenated before applying softmax
	weights = tf.Variable(
		tf.truncated_normal([vocabulary_size, concatenated_embed_vector_length],
			stddev=1.0 / math.sqrt(concatenated_embed_vector_length)))
	# softmax biases
	biases = tf.Variable(tf.zeros([vocabulary_size]))

	# Model.
	# Look up embeddings for inputs.
	# shape: (batch_size, embeddings_size)
	embed = [] # collect embedding matrices with shape=(batch_size, embedding_size)
	for j in range(window_size):
		embed_w = tf.nn.embedding_lookup(word_embeddings, train_dataset[:, j])
		embed.append(embed_w)

	embed_d = tf.nn.embedding_lookup(doc_embeddings, train_dataset[:, window_size])
	embed.append(embed_d)
	# concat word and doc vectors
	embed = tf.concat(1, embed)
	
	# Compute the loss, using a sample of the negative labels each time.
	loss = tf.nn.sampled_softmax_loss(weights, biases, embed,
		train_labels, n_neg_samples, vocabulary_size)
	loss = tf.reduce_mean(loss)

	# Optimizer.
	if optimize == 'Adagrad':
		optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(loss)
	elif optimize == 'SGD':
		optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

	# Compute the similarity between minibatch examples and all embeddings.
	# We use the cosine distance:
	norm_w = tf.sqrt(tf.reduce_sum(tf.square(word_embeddings), 1, keep_dims=True))
	normalized_word_embeddings = word_embeddings / norm_w

	norm_d = tf.sqrt(tf.reduce_sum(tf.square(doc_embeddings), 1, keep_dims=True))
	normalized_doc_embeddings = doc_embeddings / norm_d

	# Compute paragraph similarity
	valid_embeddings = tf.nn.embedding_lookup(
		normalized_doc_embeddings, valid_dataset)
	similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_doc_embeddings))

	# init op 
	init_op = tf.initialize_all_variables()
	# create a saver 
	# saver = tf.train.Saver()


## The session
n_steps = 10001

with tf.Session(graph=graph) as session:
	session.run(init_op)
	average_loss = 0
	print("Initialized")
	for step in range(n_steps):
		batch_data, batch_labels = generate_batch_pvdm(batch_size, window_size)
		feed_dict = {train_dataset : batch_data, train_labels : batch_labels}
		op, l = session.run([optimizer, loss], feed_dict=feed_dict)
		average_loss += l
		if step % 2000 == 0:
			if step > 0:
				average_loss = average_loss / 2000
			# The average loss is an estimate of the loss over the last 2000 batches.
			print('Average loss at step %d: %f' % (step, average_loss))
			average_loss = 0
		# note that this is expensive (~20% slowdown if computed every 500 steps)
		if step % 10000 == 0:
			sim = similarity.eval()
			for i in range(valid_size):
				valid_doc = ' '.join(docs[valid_examples[i]])
				top_k = 8 # number of nearest neighbors
				nearest = (-sim[i, :]).argsort()[:top_k]
				log = 'Nearest to "%s":' % valid_doc
				for k in range(top_k):
					close_doc = ' '.join(docs[nearest[k]])
					log = '%s "%s"\n' % (log, close_doc)
				print(log)

	final_embeddings_w = session.run(normalized_word_embeddings)
	final_embeddings_d = session.run(normalized_doc_embeddings)




import zipfile
from word2vec import *
# Read the data into a list of strings.
def read_data(filename):
  """Extract the first file enclosed in a zip file as a list of words"""
  with zipfile.ZipFile(filename) as f:
    data = f.read(f.namelist()[0]).split()
  return data


filename = 'data/text8.zip'
words = read_data(filename)
words = words[:10000]
print('Data size', len(words))

# Step 2: Build the dictionary and replace rare words with UNK token.
vocabulary_size = 500

# data, count, dictionary, reverse_dictionary = build_dataset(words, vocabulary_size=vocabulary_size)
# del words  # Hint to reduce memory.
# print('Most common words (+UNK)', count[:5])
# print('Sample data', data[:10])
# print(len(data))


w2v = Word2Vec(vocabulary_size=vocabulary_size, 
	architecture='cbow',
	# loss_type='nce_loss',
	n_steps=2001)

# print w2v.get_params()
w2v.fit(words)
print(w2v.final_embeddings.shape)
print(len(w2v.sort('the')))

print('words closest to %s:' % 'the')
print(w2v.sort('the')[:10])

# print([reverse_dictionary[i] for i in range(3)])
# print(w2v.transform([0,1,2,3]).shape)

save_path = w2v.save('models')
print(w2v.final_embeddings[0,0])

print save_path

# restore a saved model
w2c_restored = Word2Vec.restore(save_path)
print(w2c_restored.final_embeddings[0,0])
print(w2c_restored.dictionary['the'])
print(w2c_restored.reverse_dictionary.items()[:5])



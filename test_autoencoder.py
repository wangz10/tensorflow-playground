from autoencoders import *

import sklearn.preprocessing as prep
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets('MNIST_data', one_hot = True)

def standard_scale(X_train, X_test):
    preprocessor = prep.StandardScaler().fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_test = preprocessor.transform(X_test)
    return X_train, X_test

def get_random_block_from_data(data, batch_size):
    start_index = np.random.randint(0, len(data) - batch_size)
    return data[start_index:(start_index + batch_size)]

X_train, X_test = standard_scale(mnist.train.images, mnist.test.images)

n_samples = int(mnist.train.num_examples)
training_epochs = 5
batch_size = 128
display_step = 1

ae = BaseAutoencoder(n_input=784, n_hidden=100, logdir='logs')

for epoch in range(training_epochs):
    # avg_cost = 0.
    total_batch = int(n_samples / batch_size)
    # Loop over all batches
    for i in range(total_batch):
        batch_xs = get_random_block_from_data(X_train, batch_size)

        # Fit training using batch data
        cost = ae.partial_fit(batch_xs)
        # # Compute average loss
        # avg_cost += cost / n_samples * batch_size

    # Display logs per epoch step
    if epoch % display_step == 0:
        print "Epoch:", '%04d' % (epoch + 1), \
            "loss=", "{:.9f}".format(cost), \
            "step=%d" % i

print "Total loss: " + str(ae.calc_total_cost(X_test))
print ae.global_step

saved_path = ae.save('models/base_ae')

ae_restored = BaseAutoencoder.restore(saved_path)
print "Total loss: " + str(ae_restored.calc_total_cost(X_test))
print ae_restored.global_step





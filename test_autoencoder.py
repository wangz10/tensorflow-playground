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


ae = BaseAutoencoder(n_input=784, n_hidden=[400, 200], logdir='logs/ae', log_every_n=200)
# deep_ae = BaseAutoencoder(n_input=784, n_hidden=[300, 200], logdir='logs/deep_ae', log_every_n=200)
gae = AdditiveGaussianNoiseAutoencoder(n_input=784, n_hidden=[400, 200], logdir='logs/gae', log_every_n=200)
dae = MaskingNoiseAutoencoder(n_input=784, n_hidden=[400, 200], logdir='logs/dae', log_every_n=200)

model_classes = [BaseAutoencoder, AdditiveGaussianNoiseAutoencoder, MaskingNoiseAutoencoder]
models = [ae, gae, dae]
model_names = ['ae', 'gae', 'dae']

# model_classes = [BaseAutoencoder, BaseAutoencoder]
# models = [ae, deep_ae]
# model_names = ['ae', 'deep_ae']


for epoch in range(training_epochs):
    # avg_cost = 0.
    total_batch = int(n_samples / batch_size)
    # Loop over all batches
    for i in range(total_batch):
        batch_xs = get_random_block_from_data(X_train, batch_size)

        # Fit training using batch data
        cost1 = ae.partial_fit(batch_xs)
        # cost2 = deep_ae.partial_fit(batch_xs)
        cost2 = gae.partial_fit(batch_xs)
        cost3 = dae.partial_fit(batch_xs)

        # # Compute average loss
        # avg_cost += cost / n_samples * batch_size

    # Display logs per epoch step
    if epoch % display_step == 0:
        print "Epoch:", '%04d' % (epoch + 1), \
            "ae loss=", "{:.9f}".format(cost1), \
            "gae loss=", "{:.9f}".format(cost2), \
            "dae loss=", "{:.9f}".format(cost3)


print "Total loss ae: " + str(ae.calc_total_cost(X_test))
# print "Total loss deep_ae: " + str(deep_ae.calc_total_cost(X_test))
print "Total loss gae: " + str(gae.calc_total_cost(X_test))
print "Total loss dae: " + str(dae.calc_total_cost(X_test))

# print ae.global_step

for i in range(2):
    model_name = model_names[i]
    model = models[i]
    model_class = model_classes[i]

    saved_path = model.save('models/%s' % model_name)
    ae_restored = model_class.restore(saved_path)
    print "Total loss: " + str(ae_restored.calc_total_cost(X_test))
    print ae_restored.global_step







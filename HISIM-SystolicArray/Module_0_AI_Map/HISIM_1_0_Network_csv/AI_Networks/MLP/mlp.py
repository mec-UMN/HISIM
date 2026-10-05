""" Multilayer Perceptron.

A Multilayer Perceptron (Neural Network) implementation example using
TensorFlow library. This example is using the MNIST database of handwritten
digits (http://yann.lecun.com/exdb/mnist/).

Links:
	[MNIST Dataset](http://yann.lecun.com/exdb/mnist/).

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

# ------------------------------------------------------------------
#
# THIS EXAMPLE HAS BEEN RENAMED 'neural_network.py', FOR SIMPLICITY.
#
# ------------------------------------------------------------------


from __future__ import print_function
from hardware_estimation import hardware_estimation

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

import tensorflow as tf

# Parameters
learning_rate = 0.01
training_epochs = 1
batch_size = 64
display_step = 1

# Network Parameters
n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 128 # 2nd layer number of neurons
n_hidden_3 = 512 # 2nd layer number of neurons
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_classes])

_input_array = []
_weights = []

def batch_norm(input):
	return tf.contrib.layers.batch_norm(input, decay=0.9, center=True, scale=True, epsilon=1e-3, is_training=True, upDATE_3ds_collections=None)


# Create model
def multilayer_perceptron(x):
	# Hidden fully connected layer with 256 neurons
	w1 = tf.get_variable(shape=[n_input, n_hidden_1], initializer=tf.contrib.keras.initializers.he_normal(), name='w1')
	b1 = tf.get_variable(shape=[n_hidden_1], initializer=tf.contrib.keras.initializers.he_normal(), name='b1')
	_input_array.append(x)
	_weights.append(w1)
	layer_1 = tf.nn.relu(batch_norm(tf.add(tf.matmul(x,w1), b1)))

	# Hidden fully connected layer with 256 neurons
	w2 = tf.get_variable(shape=[n_hidden_1, n_hidden_1], initializer=tf.contrib.keras.initializers.he_normal(), name='w2')
	b2 = tf.get_variable(shape=[n_hidden_1], initializer=tf.contrib.keras.initializers.he_normal(), name='b2')
	_input_array.append(layer_1)
	_weights.append(w2)
	layer_2 = tf.nn.relu(batch_norm(tf.add(tf.matmul(layer_1,w2), b2)))

	# Hidden fully connected layer with 256 neurons
	w3 = tf.get_variable(shape=[n_hidden_1, n_hidden_2], initializer=tf.contrib.keras.initializers.he_normal(), name='w3')
	b3 = tf.get_variable(shape=[n_hidden_2], initializer=tf.contrib.keras.initializers.he_normal(), name='b3')
	_input_array.append(layer_2)
	_weights.append(w3)
	layer_3 = tf.nn.relu(batch_norm(tf.add(tf.matmul(layer_2,w3), b3)))

	# # Hidden fully connected layer with 256 neurons
	# w4 = tf.get_variable(shape=[n_hidden_2, n_hidden_3], initializer=tf.contrib.keras.initializers.he_normal(), name='w4')
	# b4 = tf.get_variable(shape=[n_hidden_3], initializer=tf.contrib.keras.initializers.he_normal(), name='b4')
	# _input_array.append(layer_3)
	# _weights.append(w4)
	# layer_4 = tf.nn.relu(batch_norm(tf.add(tf.matmul(layer_3,w4), b4)))

	# # Hidden fully connected layer with 256 neurons
	# w5 = tf.get_variable(shape=[n_hidden_3, n_hidden_1], initializer=tf.contrib.keras.initializers.he_normal(), name='w5')
	# b5 = tf.get_variable(shape=[n_hidden_1], initializer=tf.contrib.keras.initializers.he_normal(), name='b5')
	# _input_array.append(layer_4)
	# _weights.append(w5)
	# layer_5 = tf.nn.relu(batch_norm(tf.add(tf.matmul(layer_4,w5), b5)))

	# # Hidden fully connected layer with 256 neurons
	# w6 = tf.get_variable(shape=[n_hidden_1, n_hidden_2], initializer=tf.contrib.keras.initializers.he_normal(), name='w6')
	# b6 = tf.get_variable(shape=[n_hidden_2], initializer=tf.contrib.keras.initializers.he_normal(), name='b6')
	# _input_array.append(layer_5)
	# _weights.append(w6)
	# layer_6 = tf.nn.relu(batch_norm(tf.add(tf.matmul(layer_5,w6), b6)))

	# Output fully connected layer with a neuron for each class
	w_out = tf.get_variable(shape=[n_hidden_2, n_classes], initializer=tf.contrib.keras.initializers.he_normal(), name='w_out')
	b_out = tf.get_variable(shape=[n_classes], initializer=tf.contrib.keras.initializers.he_normal(), name='b_out')
	_input_array.append(layer_3)
	_weights.append(w_out)
	out_layer = tf.nn.relu(batch_norm(tf.matmul(layer_3, w_out) + b_out))
	return out_layer

# Construct model
logits = multilayer_perceptron(X)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
	logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)
# Initializing the variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)

	# Training cycle
	for epoch in range(training_epochs):
		if (epoch<20):
			learning_rate = 0.01
		else:
			learning_rate = 0.001
		avg_cost = 0.
		total_batch = int(mnist.train.num_examples/batch_size)
		# Loop over all batches
		for i in range(total_batch):
			batch_x, batch_y = mnist.train.next_batch(batch_size)
			# Run optimization op (backprop) and cost op (to get loss value)
			_, c = sess.run([train_op, loss_op], feed_dict={X: batch_x,
															Y: batch_y})
			# Compute average loss
			avg_cost += c / total_batch
		# Display logs per epoch step
		if epoch % display_step == 0:
			print("Epoch:", '%04d' % (epoch+1), "cost={:.9f}".format(avg_cost))
	print("Optimization Finished!")

	# Test model
	pred = tf.nn.softmax(logits)  # Apply softmax to logits
	correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
	# Calculate accuracy
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
	print("Accuracy:", accuracy.eval({X: mnist.test.images, Y: mnist.test.labels}))

	feed_dict={X: mnist.test.images, Y: mnist.test.labels}
	# H = _input_array.eval({X: mnist.test.images, Y: mnist.test.labels})
	# W = _weights.eval({X: mnist.test.images, Y: mnist.test.labels})
	H, W = sess.run([_input_array, _weights], feed_dict=feed_dict)
	hardware_estimation(H,W,8,8)

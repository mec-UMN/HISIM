#MNIST Structure without split
#Author: Gokul Krishnan
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile

import numpy as np
import pickle
import tensorflow as tf
import utils_mnist
import sys
import select
import os
from DATE_3dtime import DATE_3dtime
import time
from IPython import embed
#from StringIO import StringIO
import matplotlib.pyplot as plt
from operator import sub
from hardware_estimation import hardware_estimation

from tensorflow.python.client import timeline
from tensorflow.examples.tutorials.mnist import input_data


tf.app.flags.DEFINE_integer('num_classes', 10, """Number of classes in the dataset.""")
tf.app.flags.DEFINE_integer('ngroups2', 4, """Grouping number on FC""")
tf.app.flags.DEFINE_integer('ngroups1', 4, """Grouping number on Conv""")
tf.app.flags.DEFINE_float('gamma1', 2.0, """Overlap loss regularization parameter""")
tf.app.flags.DEFINE_float('gamma2', 2.0, """Weight split loss regularization parameter""")
tf.app.flags.DEFINE_float('gamma3', 2.0, """Uniform loss regularization paramter""")
tf.app.flags.DEFINE_integer('max_steps', 20000, """Number of batches to run.""")
tf.app.flags.DEFINE_string('basemodel', None, """Base model to load paramters""")
tf.app.flags.DEFINE_string('checkpoint', None, """Model checkpoint to load""")
tf.app.flags.DEFINE_float ('_weights', 0, """Wegihts in the whole model""")
tf.app.flags.DEFINE_float('_flops', 0, """The flops in the whole design""")
tf.app.flags.DEFINE_float('_total_flops', 0, """The flops in the whole design""")
tf.app.flags.DEFINE_float('_total_weights', 0, """The flops in the whole design""")
tf.app.flags.DEFINE_string('train_dir', './train/base.ckpt', """Directory where to write log and checkpoint.""")
tf.app.flags.DEFINE_float ('prob_fc1', 0, """Rewiring Probability""")
tf.app.flags.DEFINE_float ('prob_fc2', 0, """Rewiring Probability""")
tf.app.flags.DEFINE_float ('prob_conv2', 0, """Rewiring Probability""")
tf.app.flags.DEFINE_float ('prob_conv1', 0, """Rewiring Probability""")
tf.app.flags.DEFINE_float('_total_flops_1', 0, """The flops in the sw design""")
tf.app.flags.DEFINE_float('_total_weights_1', 0, """The flops in the sw design""")
tf.app.flags.DEFINE_float('_total_flops_2', 0, """The flops in the sw design""")
tf.app.flags.DEFINE_float('_total_weights_2', 0, """The flops in the sw design""")
tf.app.flags.DEFINE_float('_total_flops_3', 0, """The flops in the sw design""")
tf.app.flags.DEFINE_float('_total_weights_3', 0, """The flops in the sw design""")
tf.app.flags.DEFINE_float('_total_flops_4', 0, """The flops in the sw design""")
tf.app.flags.DEFINE_float('_total_weights_4', 0, """The flops in the sw design""")
tf.app.flags.DEFINE_float('_total_flops_sw', 0, """The flops in the sw design""")
tf.app.flags.DEFINE_float('_total_weights_sw', 0, """The flops in the sw design""")
tf.app.flags.DEFINE_float('beta_fc2', 1, """The flops in the sw design""")
tf.app.flags.DEFINE_float('beta_fc1', 1, """The flops in the sw design""")
tf.app.flags.DEFINE_float('beta_conv2', 1, """The flops in the sw design""")
tf.app.flags.DEFINE_float('beta_conv1', 1, """The flops in the sw design""")
tf.app.flags.DEFINE_float('learning_rate', 1e-3, """The flops in the sw design""")

FLAGS1 = tf.app.flags.FLAGS
tf.logging.set_verbosity(tf.logging.INFO)


def pickle_params(self, file_name_string, param_name):
	file = open(file_name_string, 'wb')   # file_name_string = 'allParameters_0.pickle'
	pickle.dump(param_name, file)
	file.close()

def _add_flops_weights(scope_name, f, w ):

	#if scope_name not in _counted_scope:
	FLAGS1._flops = 0
	FLAGS1._weights = 0
	FLAGS1._flops += f
	FLAGS1._total_flops += FLAGS1._flops
	FLAGS1._weights += w
	FLAGS1._total_weights += FLAGS1._weights
	print("\nThe number of flops for layer %s is:%g\n" %(scope_name, (FLAGS1._flops)))
	print("\n The number of weights for layer %s is: %g \n" %(scope_name, (FLAGS1._weights)))


def deepnn(x):
	# Reshape to use within a convolutional neural net.
	# Last dimension is for "features" - there is only one here, since images are
	# grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.

	print('\tBuilding unit: MNIST Structure')

	_input_array = []
	_weights = []

	with tf.name_scope('reshape'):
	  x_image = tf.reshape(x, [-1, 28, 28, 1])
	  print("The shape of the input image is", x_image.get_shape())

	  _input_array.append(tf.transpose(x_image, (0,3,1,2)))

	# First convolutional layer - maps one grayscale image to 32 feature maps.
	with tf.name_scope('conv1'):
		b, h, w, in_channel = x_image.get_shape().as_list()
		W_conv1 = weight_variable([5, 5, 1, 20])
		_weights.append(W_conv1)
		b_conv1 = bias_variable([20])
		f = 2 * (h) * (w) * 1 * 20 * 5* 5
		#print ('\n f is ', f)
		w = 1 * 32 * 5* 5
		scope_name = 'conv1'
		_add_flops_weights(scope_name, f, w)
		h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
		print("The shape of the first convolutional output is", h_conv1.get_shape())

	# Pooling layer - downsamples by 2X.
	with tf.name_scope('pool1'):
		h_pool1 = max_pool_2x2(h_conv1)
		print("The shape of the first pooling output is", h_pool1.get_shape())


	_input_array.append(tf.transpose(h_pool1, (0,3,1,2)))

	# Second convolutional layer -- maps 32 feature maps to 64.
	with tf.name_scope('conv2'):
		b, h, w, in_channel = x_image.get_shape().as_list()
		W_conv2 = weight_variable([5, 5, 20, 50])
		_weights.append(W_conv2)
		b_conv2 = bias_variable([50])
		f = 2 * (h) * (w) * 20 * 50 * 5* 5
		#print ('\n f is ', f)
		w = 20 * 50 * 5* 5
		scope_name = 'conv2'
		_add_flops_weights(scope_name, f, w)
		h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
		print("The shape of the Second convolutional output is", h_conv2.get_shape())

	# Second pooling layer.
	with tf.name_scope('pool2'):
		h_pool2 = max_pool_2x2(h_conv2)
		print("The shape of the first pooling output is", h_pool2.get_shape())


	# Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
	# is down to 7x7x64 feature maps -- maps this to 1024 features.
	with tf.name_scope('fc1'):
		W_fc1 = weight_variable([7 * 7 * 50, 500])
		_weights.append(W_fc1)
		in_shape, out_shape = W_fc1.get_shape().as_list()
		#print("\n\nthe size of the randommizer is %d, %d" %(s1, s2))
		b_fc1 = bias_variable([500])
		h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 50])
		_input_array.append(h_pool2_flat)
		#rand = utils_sw.wrapper(W_fc1, out_shape, (in_shape+out_shape), FLAGS1.prob, in_shape, out_shape)
		#s1, s2 = rand.get_shape().as_list()
		#W_fc1_sw = tf.multiply(W_fc1,rand, name="weights_smw")
		f = 2 * (3200 + 1) * 500
		w = (3200 + 1) * 500
		scope_name = 'fc1'
		_add_flops_weights(scope_name, f, w)
		h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)



	# Dropout - controls the complexity of the model, prevents co-adaptation of
	# features.
	with tf.name_scope('dropout'):
		keep_prob = tf.placeholder(tf.float32)
		h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

	_input_array.append(h_fc1_drop)

	# Map the 1024 features to 10 classes, one for each digit
	with tf.name_scope('fc2'):
		W_fc2 = weight_variable([500, 10])
		_weights.append(W_fc1)
		b_fc2 = bias_variable([10])
		#y_conv = _fc_1(h_fc1_drop, 10, input_q=split_p2, output_q=split_q2, name = 'fc2')
		f = 2 * (500 + 1) * 10
		w = (500 + 1) * 10
		scope_name = 'fc2'
		_add_flops_weights(scope_name, f, w)
		y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


	return y_conv, keep_prob, _weights, _input_array

def conv2d(x, W):
	"""conv2d returns a 2d convolution layer with full stride."""
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
	"""max_pool_2x2 downsamples a feature map by 2X."""
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
						  strides=[1, 2, 2, 1], padding='SAME')

def weight_variable(shape):
	"""weight_variable generates a weight variable of a given shape."""
	with tf.name_scope('weights'):
		initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	"""bias_variable generates a bias variable of a given shape."""
	with tf.name_scope('biases'):
		initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)


def main(_):
	# Import data
	mnist = input_data.read_data_sets(FLAGS.data_dir)
	global FLAGS1

	_input_array = []
	_weights = []

	# Create the model
	x = tf.placeholder(tf.float32, [None, 784])

	# Define loss and optimizer
	y_ = tf.placeholder(tf.int64, [None])

	# Build the graph for the deep net
	y_conv, keep_prob, _weights, _input_array = deepnn(x)

	cross_entropy = tf.losses.sparse_softmax_cross_entropy(
		labels=y_, logits=y_conv)

	total_loss = cross_entropy

	l2 = tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
	total_loss = cross_entropy + l2 * 0.0003

	with tf.name_scope('adam_optimizer'):
		train_step = tf.train.AdamOptimizer(FLAGS1.learning_rate).minimize(cross_entropy)

	with tf.name_scope('accuracy'):
		correct_prediction = tf.equal(tf.argmax(y_conv, 1), y_)
		correct_prediction = tf.cast(correct_prediction, tf.float32)
		accuracy = tf.reduce_mean(correct_prediction)

	graph_location = tempfile.mkdtemp() #Creates a temporary file
	print('Saving graph to: %s' % graph_location)
	train_writer = tf.summary.FileWriter(graph_location)  #Creates an event file
	train_writer.add_graph(tf.get_default_graph())

	allParameters_before = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
	saver = tf.train.Saver(tf.global_variables(), max_to_keep=10000)
	#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
	#with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
	begin = time.time()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		first_trainable_var_bf = sess.run(allParameters_before)  # save weight before training
		for i in range(FLAGS1.max_steps):
			batch = mnist.train.next_batch(50)
			if i % 100 == 0:
				train_accuracy = accuracy.eval(feed_dict={
					x: batch[0], y_: batch[1], keep_prob: 1.0})
				print('step %d, training accuracy %g' % (i, train_accuracy))
			train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})


		# compute in batches to avoid OOM on GPUs
		accuracy_l = []
		save_path = saver.save(sess, FLAGS1.train_dir)
		for _ in range(20):
			batch = mnist.test.next_batch(500, shuffle=True)
			accuracy_l.append(accuracy.eval(feed_dict={x: batch[0],
													   y_: batch[1],
													   keep_prob: 1.0}))
		print('test accuracy %g' % np.mean(accuracy_l))
		print("\nThe number of flops is:%g\n" %((FLAGS1._total_flops)))
		print("\n The number of weights is: %g \n" %((FLAGS1._total_weights)))

		print("*******************Invoking NeuroSim*********************")


		#batch = mnist.test.next_batch(500, shuffle=True)
		#feed_dict = {x: batch[0], y_: batch[1], keep_prob: 1.0}
		#H, W = sess.run([_input_array, _weights], feed_dict=feed_dict)
		#hardware_estimation(H,W,8,8)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
					  default='/tmp/tensorflow/mnist/input_data',
					  help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

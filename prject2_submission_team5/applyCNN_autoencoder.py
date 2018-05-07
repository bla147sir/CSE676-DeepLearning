import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data
import generatePickle
import autoencoder_ageest
# mnist = input_data.read_data_sets("/tmp/data", one_hot = True)

#http://www.ritchieng.com/machine-learning/deep-learning/tensorflow/regularization/

n_classes = 100
batch_size = 32

x = tf.placeholder('float',[None, 10000])
y = tf.placeholder('float')

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def deconv2d(x, W, output_shape):
	return tf.nn.conv2d_transpose(x, W, strides=[1,1,1,1], padding='SAME')

def maxpool2d(x):
	return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def conv_neural_network(x):
	weights = {
				'W_conv1': tf.Variable(tf.random_normal([5,5,1,32])),
				'W_conv2': tf.Variable(tf.random_normal([5,5,32,64])),
				'W_fc': tf.Variable(tf.random_normal([25*25*64,1024])),  #this 1024 might be small?
				'output': tf.Variable(tf.random_normal([1024,n_classes]))
			  }

	biases = {
				'b_conv1': tf.Variable(tf.random_normal([32])),
				'b_conv2': tf.Variable(tf.random_normal([64])),
				'b_fc': tf.Variable(tf.random_normal([1024])),
				'output': tf.Variable(tf.random_normal([n_classes]))
			 }

	x = tf.reshape(x, [-1,100,100,1])

	conv1 = tf.nn.relu(conv2d(x, weights['W_conv1'])+biases['b_conv1'])
	conv1 = maxpool2d(conv1)

	conv2 = tf.nn.relu(conv2d(conv1, weights['W_conv2'])+biases['b_conv2'])
	conv2 = maxpool2d(conv2)

	fc = tf.reshape(conv2, [-1, 25*25*64])
	fc = tf.nn.relu(tf.matmul(fc, weights['W_fc'])+biases['b_fc'])

	output = tf.matmul(fc, weights['output'])+biases['output']

	return output

def get_next_batch(dataset, batch_size, batch_num):
	return dataset[batch_num*batch_size : (batch_num+1)*batch_size]

def train_neural_network(x, train_x, train_y, test_x, test_y):
	prediction = conv_neural_network(x)

	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))

	optimizer = tf.train.AdamOptimizer().minimize(cost)


	config = tf.ConfigProto()
	config.gpu_options.allocator_type = 'BFC'
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		hm_epoch = 10
		for epoch in range(hm_epoch):
			epoch_loss = 0
			for _ in range(int(len(train_x)/batch_size)):
				# epoch_x, epoch_y = mnist.train.next_batch(batch_size)
				epoch_x = get_next_batch(train_x, batch_size, _)
				epoch_y = get_next_batch(train_y, batch_size, _)
				_, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
				epoch_loss += c

			print('Epoch', epoch+1, 'completed out of', hm_epoch, 'loss:', epoch_loss)

		# correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
		correct = tf.less_equal(tf.argmax(prediction, 1), tf.argmax(y, 1), 3)
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

		# n_batches = mnist.test.images.shape[0] // 50
		n_batches = len(test_x) // batch_size
		cumulative_accuracy = 0

		for _ in range(n_batches):
			# test_x, test_y = mnist.test.next_batch(50)
			current_x = get_next_batch(test_x, batch_size, _)
			current_y = get_next_batch(test_y, batch_size, _)
			cumulative_accuracy += accuracy.eval(feed_dict = {x: current_x, y: current_y})

		#print('Accuracy:', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
		print('Accuracy:', cumulative_accuracy/n_batches)

# train_neural_network(x)

if __name__ == '__main__':
	mat_dire = '/home/jiawei_zhao/wiki_crop/wiki.mat'
	img_dire = '/home/jiawei_zhao/wiki_crop/'

	train_x, train_y, test_x, test_y, autoencode_y = generatePickle.generateData(mat_dire, img_dire)
	autoencoder_ageest.autoencode(train_x, test_x, autoencode_y)

	train_x = autoencoder_ageest.encodeData(train_x)
	test_x = autoencoder_ageest.encodeData(test_x)

	train_neural_network(x, train_x, train_y, test_x, test_y)



from __future__ import division, print_function, absolute_import

import matplotlib
matplotlib.use('Agg')
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Visualize encoder setting
# Parameters
learning_rate = 0.01    # 0.01 this learning rate will be better! Tested
training_epochs = 20
batch_size = 64
display_step = 1
# Network Parameters
n_input = 100*100  # MNIST data input (img shape: 28*28)
# tf Graph input (only pictures)
X = tf.placeholder("float", [None, n_input])

# network setup for visualization
n_hidden_1 = 2048
n_hidden_2 = 1024
n_hidden_3 = 128
n_hidden_4 = 10
n_hidden_5 = 2
'''
# network setup for preprocess
n_hidden_1 = 2048
n_hidden_2 = 1024
n_hidden_3 = 512
n_hidden_4 = 256
n_hidden_5 = 128
'''

weights = {
    'encoder_h1': tf.Variable(tf.truncated_normal([n_input, n_hidden_1],)),
    'encoder_h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2],)),
    'encoder_h3': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_3],)),
    'encoder_h4': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_4],)),
    'encoder_h5': tf.Variable(tf.truncated_normal([n_hidden_4, n_hidden_5],)),
    'decoder_h1': tf.Variable(tf.truncated_normal([n_hidden_5, n_hidden_4],)),
    'decoder_h2': tf.Variable(tf.truncated_normal([n_hidden_4, n_hidden_3],)),
    'decoder_h3': tf.Variable(tf.truncated_normal([n_hidden_3, n_hidden_2],)),
    'decoder_h4': tf.Variable(tf.truncated_normal([n_hidden_2, n_hidden_1],)),
    'decoder_h5': tf.Variable(tf.truncated_normal([n_hidden_1, n_input],))
}

biases = {
    'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'encoder_b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'encoder_b4': tf.Variable(tf.random_normal([n_hidden_4])),
    'encoder_b5': tf.Variable(tf.random_normal([n_hidden_5])),
    'decoder_b1': tf.Variable(tf.random_normal([n_hidden_4])),
    'decoder_b2': tf.Variable(tf.random_normal([n_hidden_3])),
    'decoder_b3': tf.Variable(tf.random_normal([n_hidden_2])),
    'decoder_b4': tf.Variable(tf.random_normal([n_hidden_1])),
    'decoder_b5': tf.Variable(tf.random_normal([n_input])),
}

def encoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['encoder_h3']),
                                   biases['encoder_b3']))
    layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, weights['encoder_h4']),
                                   biases['encoder_b4']))
    layer_5 = tf.add(tf.matmul(layer_4, weights['encoder_h5']),
                                    biases['encoder_b5'])
    return layer_5

def decoder(x):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['decoder_h3']),
                                biases['decoder_b3']))
    layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, weights['decoder_h4']),
                                biases['decoder_b4']))
    layer_5 = tf.nn.sigmoid(tf.add(tf.matmul(layer_4, weights['decoder_h5']),
                                   biases['decoder_b5']))
    return layer_5

def get_next_batch(x, batch_size, batch_num):
    return x[batch_num*batch_size : (batch_num+1)*batch_size]

def autoencode(data, testData, testLabel):
    # Construct model
    encoder_op = encoder(X)
    decoder_op = decoder(encoder_op)

    # Prediction
    y_pred = decoder_op
    # Targets (Labels) are the input data.
    y_true = X

    # Define loss and optimizer, minimize the squared error
    cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    # Launch the graph
    with tf.Session() as sess:
        # tf.initialize_all_variables() no long valid from
        # 2017-03-02 if using tensorflow >= 0.12
        init = tf.global_variables_initializer()
        sess.run(init)
        total_batch = int(len(data)/batch_size)
        # Training cycle
        for epoch in range(training_epochs):
            # Loop over all batches
            for i in range(total_batch):
                batch_xs = get_next_batch(data, batch_size, i)  # max(x) = 1, min(x) = 0
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1),
                      "cost=", "{:.9f}".format(c))

        print("Optimization Finished!")

        encoder_result = sess.run(encoder_op, feed_dict={X: testData})
        # testLabel = np.asarray(testLabel)
        plt.scatter(encoder_result[:, 0], encoder_result[:, 1], c=testLabel)
        plt.colorbar()
        plt.savefig('encode_res.png')

def encodeData(data):
    # Construct model
    encoder_op = encoder(X)
    decoder_op = decoder(encoder_op)

    # Prediction
    y_pred = decoder_op
    # Targets (Labels) are the input data.
    y_true = X

    cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        total_batch = int(len(data)/batch_size)
        
        for epoch in range(training_epochs):
            for i in range(total_batch):
                batch_xs = get_next_batch(data, batch_size, i)  # max(x) = 1, min(x) = 0
                _, c = sess.run([optimizer, cost], feed_dict={X: batch_xs})
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1),
                      "cost=", "{:.9f}".format(c))

        print("Optimization Finished!")

        encode_decode = sess.run(
            y_pred, feed_dict={X: data})
        
        return encode_decode



from __future__ import print_function

import os

import tensorflow as tf
from tensorflow.contrib import rnn

import config
from utils.data_loader import get_cancer_data_as_object
from utils.file_name_generator import get_filename
from utils.statistics import MLResult


data_file_url = os.path.join(config.BASE_DIR+  '/data/breast_cancer.csv')
cancer_patients = get_cancer_data_as_object(data_file_url, output_column_index=1 )


# Training Parameters
learning_rate = 0.001
training_steps = 20
batch_size = 128
display_step = 10

# Network Parameters
num_input = 30
timesteps = 1 # timesteps
num_hidden = 256 # hidden layer num of features
num_classes = 2  # 4 # total classes (0-3 digits)

# tf Graph input
X = tf.placeholder("float", [None, timesteps, num_input], name="X")
Y = tf.placeholder("float", [None, num_classes], name="Y")

# Define weights
weights = tf.Variable(tf.random_normal([num_hidden, num_classes]), name="weights")
biases = tf.Variable(tf.random_normal([num_classes]), name="biases")


# source: https://github.com/aymericdamien/TensorFlow-Examples/blob/master/examples/3_NeuralNetworks/recurrent_network.py
def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    #
    #
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, timesteps, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights) + biases

logits = RNN(X, weights, biases)
prediction = tf.nn.softmax(logits)



cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=prediction)
loss_op = tf.reduce_mean(cross_entropy)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))




# Initialize the variables (i.e. assign their default value)
# init = tf.global_variables_initializer()
epochs = 10

saver = tf.train.Saver()
# Start training

with tf.Session() as sess:

    # Run the initializer
    # sess.run(init)

    try:
        saver = tf.train.import_meta_graph('Checkpoint/saved_checkpoint-' + str(epochs) + '.meta')
        # saver.restore(sess, tf.train.latest_checkpoint('Checkpoint/'))
        saver.restore(sess, "Checkpoint/saved_checkpoint-" + str(epochs))
        print("The model is restored.")
    except:

        # Initializing the variables
        print("Initializing the variables.")
        # sess.run(init)
        sess.run(tf.global_variables_initializer())

    costs = []
    for count in range(0, epochs):
        total_batch_size = int(cancer_patients.train.num_examples/batch_size)
        costs_in_each_epochs = []
        for step in range(1, training_steps):
            batch_x, batch_y = cancer_patients.train.next_batch(batch_size)
            # Reshape data to get 28 seq of 28 elements
            batch_x = batch_x.reshape((batch_size, timesteps, num_input))
            # Run optimization op (backprop)
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
            if step % display_step == 0 or step == 1:
                # Calculate batch loss and accuracy
                # sess.run([loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y})
                loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y})
                costs_in_each_epochs.append(loss)
                print("Step " + str(step) + ", Minibatch Loss= " + \
                      "{:.4f}".format(loss) + ", Training Accuracy= " + \
                      "{:.3f}".format(acc))
        costs.append(tf.reduce_mean(costs_in_each_epochs).eval())



    print("Optimization Finished!")

    test_data = cancer_patients.test.patient_profiles.reshape((-1, timesteps, num_input))
    test_label = cancer_patients.test.labels
    # saver.restore(sess, "/tmp/model.ckpt")
    print("******Testing Accuracy: after epoch: ", count, "=",
          sess.run(accuracy, feed_dict={X: test_data, Y: test_label}))

    final_accuracy = sess.run(accuracy, feed_dict={X: test_data, Y: test_label})
    # publishing result

    # publishing result

    labels = ['Malignant', 'benign']
    hyper_parameters = {'learning_rate': 0.0011}
    hyper_parameters = {'learning_rate': 0.001}

    statistics = MLResult(
        hyper_parameters=str(hyper_parameters),
        labels=labels,
        accuracy=str(final_accuracy))

    # timestamp_score
    filename = 'results/' + get_filename(final_accuracy)
    statistics.toJSONInFile(filepath=filename)
    # sess.close()

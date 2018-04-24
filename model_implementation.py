import numpy as np
import tensorflow as tf
from scipy import io as spio
import time

emnist = spio.loadmat("dataset/matlab/emnist-letters.mat")

# load training dataset
x_train = emnist["dataset"][0][0][0][0][0][0]
x_train = x_train.astype(np.float32)

# load training labels
y_train = emnist["dataset"][0][0][0][0][0][1]

# load test dataset
x_test = emnist["dataset"][0][0][1][0][0][0]
x_test = x_test.astype(np.float32)

# load test labels
y_test = emnist["dataset"][0][0][1][0][0][1]

# store labels for visualization
train_labels = y_train
test_labels = y_test

# normalize
x_train /= 255
x_test /= 255

# reshape using matlab order
x_train = x_train.reshape(x_train.shape[0], 784, order="A")
x_test = x_test.reshape(x_test.shape[0], 784, order="A")

y_train = y_train.reshape(y_train.shape[0])
n_values = np.max(y_train) + 1

y_train = np.eye(n_values)[y_train]

y_test = y_test.reshape(y_test.shape[0])
n_val_test = np.max(y_test) + 1
y_test = np.eye(n_values)[y_test]

# PLACEHOLDERS
x = tf.placeholder(tf.float32, shape=[None,784])

# VARIABLES
W = tf.Variable(tf.zeros([784,27]))
b = tf.Variable(tf.zeros([27]))

#CREATE GRAPH OPERATIONS
y = tf.matmul(x,W) + b

# LOSS FUNCTION
y_true = tf.placeholder(tf.float32,[None,27])

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true,logits=y))

# OPTIMIZER
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
train = optimizer.minimize(cross_entropy)

# CREATE SESSION
init = tf.global_variables_initializer()

saver = tf.train.Saver()

print("Training Model...")

start = time.time()
with tf.Session() as sess:
    sess.run(init)
    print(".")
    for step in range(1000):
        sess.run(train,feed_dict={x:x_train,y_true:y_train})
    #Save Model
    save_path = saver.save(sess, "saved_models/one_layer_model.ckpt")

    #Evaluate Model
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_true,1))
    acc = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    print(sess.run(acc,feed_dict={x:x_test,y_true:y_test}))
end = time.time()

time_elapsed = end - start
print("Time Elapsed: " + str(time_elapsed/60) + "minutes")

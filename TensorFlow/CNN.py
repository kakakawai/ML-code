# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

#train data
mnist = input_data.read_data_sets("MNIST_data",one_hot=True)
#mnist.train (60000,784)

#input
with tf.name_scope("Input"):
    with tf.name_scope("image"):
        x = tf.placeholder(tf.float32,shape=[None,784])
    with tf.name_scope("label"):
        y_ = tf.placeholder(tf.float32,shape=[None,10])

#model parameters
with tf.name_scope("KeepProb"):
    keep_prob = tf.placeholder(tf.float32)

def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

#convolution
def conv2d(x,w):
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding="SAME")

#pooling
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

#first Con-ReLU-Pooling-layer
with tf.name_scope("Reshape"):
    x_image = tf.reshape(x,[-1,28,28,1])
with tf.name_scope("Conv1"):
    w_conv1 = weight_variable([5,5,1,32])
    b_conv1 = bias_variable([32])

    h_conv1 = tf.nn.relu(conv2d(x_image,w_conv1)+b_conv1)

with tf.name_scope("Pool1"):
    h_pool1 = max_pool_2x2(h_conv1)

#second Con-ReLU-Pooling-layer
with tf.name_scope("Conv2"):
    w_conv2 = weight_variable([5,5,32,64])
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1,w_conv2)+b_conv2)

with tf.name_scope("Pool2"):
    h_pool2 = max_pool_2x2(h_conv2)

#first Full-Connected layer
with tf.name_scope("fc1"):
    w_fc1 = weight_variable([7*7*64,1024])
    b_fc1 = bias_variable([1024])

    h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,w_fc1)+b_fc1)

#Dropout
with tf.name_scope("DropOut"):
    h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)
#model input

#output
with tf.name_scope("fc2"):
    w_fc2 = weight_variable([1024,10])
    b_fc2 = bias_variable([10])
    y_conv = tf.matmul(h_fc1_drop,w_fc2)+b_fc2

#loss
with tf.name_scope("loss"):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y_conv))
    tf.summary.scalar("Loss",cross_entropy)
#optimizer
with tf.name_scope("Adam"):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

with tf.name_scope("Accuracy"):
    correction_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correction_prediction,tf.float32))
    tf.summary.scalar("Accuracy",accuracy)
#training


with tf.Session() as sess:
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('F:/TensorLogs', sess.graph)

    sess.run(tf.global_variables_initializer())
    for i in range(50):
        batch = mnist.train.next_batch(50)
        '''
        a,b,c,d,e,f = sess.run(fetches=[h_conv1,h_pool1,h_conv2,h_pool2,h_fc1,y_conv],feed_dict={x:batch[0],y_:batch[1],keep_prob:0.5})
        print(np.shape(a))
        print(np.shape(b))
        print(np.shape(c))
        print(np.shape(d))
        print(np.shape(e))
        print(np.shape(f))
        '''

        train_accuracy = accuracy.eval(feed_dict={x:batch[0],y_:batch[1],keep_prob:1.0})
        print("step %d,training accuracy %g"%(i,train_accuracy))
        summary,acc = sess.run([merged,train_step],feed_dict={x:batch[0],y_:batch[1],keep_prob:0.5})
        writer.add_summary(summary,i)
        #train_step.run(feed_dict={x:batch[0],y_:batch[1],keep_prob:0.5})

    # evaluate
    #print("test accuracy %g"%accuracy.eval(feed_dict={x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0}))







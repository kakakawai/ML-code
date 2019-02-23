# -*- coding:utf-8 -*-

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#train data
mnist = input_data.read_data_sets("MNIST_data",one_hot=True)
sess = tf.InteractiveSession()

#parameters
with tf.name_scope("Wight"):
    w = tf.Variable(tf.zeros([784,10]))

with tf.name_scope("Bias"):
    b = tf.Variable(tf.zeros([10]))

sess.run(tf.global_variables_initializer())

#input
with tf.name_scope("Input"):
    with tf.name_scope("X"):
        x = tf.placeholder(tf.float32,shape=[None,784])
    with tf.name_scope("Y"):
        y_ = tf.placeholder(tf.float32,shape=[None,10])




with tf.name_scope("Train"):
    # output
    with tf.name_scope("WxPlusB"):
        y = tf.matmul(x, w)+b
    # loss
    with tf.name_scope("CrossEntropy"):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
        tf.summary.scalar("Loss", loss)
    # optimizer
    with tf.name_scope("GradientDescent"):
        train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)

#evaluate
with tf.name_scope("Evaluate"):
    with tf.name_scope("CorrectionPreidiction"):
        correction_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    with tf.name_scope("Accuracy"):
        accuracy = tf.reduce_mean(tf.cast(correction_prediction,tf.float32))
    tf.summary.scalar("Accuracy",accuracy)

merged = tf.summary.merge_all()
writer = tf.summary.FileWriter('F:/TensorLogs',sess.graph)

#train loop
for i in range(1000):
    if i % 10 == 0:
        summary, acc = sess.run([merged, accuracy], feed_dict={x: mnist.test.images, y_: mnist.test.labels})
        writer.add_summary(summary,i)
        print("[+]Accuarcy at step %s:%s"%(i,acc))
    else:
        batch = mnist.train.next_batch(100)
        sess.run(train_step,feed_dict={x: batch[0], y_: batch[1]})
        #train_step.run(feed_dict={x: batch[0], y_: batch[1]})



#print(accuracy.eval(feed_dict={x:mnist.test.images,y_:mnist.test.labels}))

#print()

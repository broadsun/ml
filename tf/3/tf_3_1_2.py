#coding=utf8
import tensorflow as tf
#tf.Graph生成新的计算图，不同计算图上的tensor和flow不会共享
g1 = tf.Graph()
with g1.as_default():
    v = tf.get_variable("v", initializer = tf.zeros_initializer()(shape=[2]))

with tf.Session(graph=g1) as sess:
    tf.global_variables_initializer().run()
    with tf.variable_scope("", reuse=True):
        print(sess.run(tf.get_variable("v")))

g2=tf.Graph()
with g2.as_default():
    v = tf.get_variable("v", initializer=tf.ones_initializer()(shape=[2]))
with tf.Session(graph=g2) as sess:
    tf.global_variables_initializer().run()
    with tf.variable_scope("", reuse=True):
        print(sess.run(tf.get_variable("v")))

#tf.Graph.device指定运算设备
g=tf.Graph()
g.device('/gpu:0')
with g.as_default():
    v = tf.get_variable("v", initializer=tf.ones_initializer()(shape=[2]))
with tf.Session(graph=g) as sess:
    tf.global_variables_initializer().run()
    with tf.variable_scope("", reuse=True):
         print(sess.run(tf.get_variable("v")))


print(tf.GraphKeys.GLOBAL_VARIABLES)
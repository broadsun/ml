#coding=utf8
import tensorflow as tf
#变脸的作用就是保存和更新神经网络中的参数
#变量的初始化方法
a1 = tf.random_normal([2,3], stddev=2)
a2 = tf.truncated_normal([2,3], mean=0, stddev=2)
a3 = tf.random_uniform([2,3])
a4 = tf.random_gamma([2,3], alpha=2.14)
a5 = tf.zeros([2,3])
a6 = tf.ones([2,3])
a7 = tf.fill([2,3], 9)
a8 = tf.constant([2,3])

weights = tf.Variable(tf.constant([12,13]))
w2 = tf.Variable(weights.initialized_value()*2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(w2.eval())

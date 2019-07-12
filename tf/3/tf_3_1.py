#coding=utf8
import tensorflow as tf
a = tf.constant([1.0, 2.0], name="a")
b = tf.constant([2.0, 3.0], name="b")
result = a + b
print(a.graph) #查看a的所属计算图
print(tf.get_default_graph()) #查看当前默认的计算图



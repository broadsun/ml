#coding=utf8
import tensorflow as tf

for i in tf.GraphKeys.GLOBAL_VARIABLES:
    print(i)
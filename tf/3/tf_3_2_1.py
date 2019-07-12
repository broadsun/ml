#coding=utf8
import tensorflow as tf
# a=tf.constant([1.0, 2.0], name="a")
# b=tf.constant([2.0,3.0], name="b")
# result=tf.add(a,b, name="add")
# print(result)
#Tensor("add:0", shape=(2,), dtype=float32)
#张量保存的是名字，维度，类型。张量来自节点的第几个输出


#类型一定要匹配
a=tf.constant([1, 2], name="a", dtype=tf.float32) #只有加上dtype才没有问题
b=tf.constant([2.0,3.0], name="b")
result=tf.add(a,b, name="add")
print(result)
#tf的类型，
#tf.float32, tf.float64, tf.int8, tf.int16, tf.int32, tf.int64, tf.uint8, tf.bool, tf.complex64, tf.complex128
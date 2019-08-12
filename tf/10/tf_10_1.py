#coding=utf8
import tensorflow as tf
a = tf.constant([1.0, 2.0, 3.0],shape=[3], name='a')
print("a=",a)
b = tf.constant([1.0, 2.0, 3.0],shape=[3], name='b')
print("b=",b)
c = a + b
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    print(sess.run(c))
#在配置好GPU的机器上，tf会优先在GPU上运行
#可以通过tf.device制定在哪里运行
# with tf.device('/cpu:0'):
#     a1 = tf.constant([1.0, 2.0, 3.0],shape=[3], name='a')
#     b1 = tf.constant([1.0, 2.0, 3.0],shape=[3], name='b')
# with tf.device('/gpu:0'):
#     c1 = a1 + b1

#不同版本tf对GPU的支持不同，可能不是所有的运算都能在GPU上运行
#制定allow_soft_placement参数，增加代码的可移植性，如果运算无法在GPU上运行，则自动放到CPU上
a_cpu = tf.Variable(0, name = 'a_cpu')
with tf.device('/gpu:0'):
    a_gpu = tf.Variable(0, name='a_gpu')
sess = tf.Session(config=ConfigProto(allow_soft_placement=True, log_device_placement=True))
sess.run(tf.global_variables_initializer())

#将密集型运算放在GPU上，GPU是机器中相对独立的资源，将自算放入或转出GPU需要额外的时间

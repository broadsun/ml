#coding=utf8
import tensorflow as tf

y_ = tf.constant([1.0, 0.0, 1.0])
y = tf.constant([0.9, 0.1, 0.8])
cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))
#clip_by_value把值限制在一定范围之内
v1 = tf.constant([[1.0, 2.0],[3.0, 4.0]])
v2 = tf.constant([[5.0, 6.0],[7.0, 8.0]])

v1_v2 = v1*v2
v1_m_v2 = tf.matmul(v1, v2)
r_m = tf.reduce_mean(v1)

sess = tf.InteractiveSession()
print(cross_entropy.eval())
print(v1_v2.eval())
print(v1_m_v2.eval())
print(r_m.eval())
sess.close()    
#coding=utf8
import tensorflow as tf
#tensorflow中通过变量名字获取变量的机制主要是通过tf.get_variable 和tf.variable_scope函数实现
#tf.get_variable创建和获取标量。创建变量时和 tf.Variable是等价的
print(tf.get_variable_scope().reuse)
v = tf.Variable(tf.constant(1.0, shape=[1]), name='v')
v = tf.get_variable('v', shape=[1], initializer=tf.constant_initializer(1.0))

#变量初始化函数
'''
tf.constant_initializer()
tf.random_normal_initializer()
tf.truncated_normal_initializer()
tf.random_uniform_initializer()
tf.uniform_unit_scaling_initializer()
tf.zeros_initializer()
tf.ones_initializer()
'''

# with tf.variable_scope("foo"):
#     v = tf.get_variable("v", [1], initializer=tf.constant_initializer([3.14]))

# with tf.variable_scope("foo", reuse=True):
#     v = tf.get_variable("v", [1],initializer=tf.constant_initializer([3.14]))
#     #v = tf.Variable(tf.constant(1.2))


# with tf.variable_scope("root"):
#     print(tf.get_variable_scope().reuse)
#     with tf.variable_scope("foo", reuse=True):
#         print(tf.get_variable_scope().reuse)
#         with tf.variable_scope("Bar"):
#             print(tf.get_variable_scope().reuse)
#     print(tf.get_variable_scope().reuse)


v1 = tf.get_variable("v1", [1])
print(v1.name)

with tf.variable_scope("foo"):
    v2 = tf.get_variable("v", [1])
    print(v2.name)
with tf.variable_scope("foo"):
    with tf.variable_scope("bar"):
        v3 = tf.get_variable("v", [1])
        print(v3.name)
    v4 = tf.get_variable("v1", [1])
    print(v4.name)

with tf.variable_scope("",reuse=True):
    v5 = tf.get_variable("foo/bar/v", [1])
    print(v5 == v3)
    v6 = tf.get_variable("v1", [1])     
    print(v6 == v1)



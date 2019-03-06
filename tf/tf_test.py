import tensorflow as tf
a = tf.constant([1.0, 2.0], name="a")
b = tf.constant([2.0, 3.0], name="b")
result = a+b
print(a.graph)
print(tf.get_default_graph())
print(result)
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

g=tf.Graph()
g.device('/gpu:0')
with g.as_default():
    v = tf.get_variable("v", initializer=tf.ones_initializer()(shape=[2]))
with tf.Session(graph=g) as sess:
    tf.global_variables_initializer().run()
    with tf.variable_scope("", reuse=True):
        print(sess.run(tf.get_variable("v")))









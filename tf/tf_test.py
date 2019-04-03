# import tensorflow as tf
# a = tf.constant([1.0, 2.0], name="a")
# b = tf.constant([2.0, 3.0], name="b")
# result = a+b
# print(a.graph)
# print(tf.get_default_graph())
# print(result)
# g1 = tf.Graph()
# with g1.as_default():
#     v = tf.get_variable("v", initializer = tf.zeros_initializer()(shape=[2]))

# with tf.Session(graph=g1) as sess:
#     tf.global_variables_initializer().run()
#     with tf.variable_scope("", reuse=True):
#         print(sess.run(tf.get_variable("v")))

# g2=tf.Graph()
# with g2.as_default():
#     v = tf.get_variable("v", initializer=tf.ones_initializer()(shape=[2]))
# with tf.Session(graph=g2) as sess:
#     tf.global_variables_initializer().run()
#     with tf.variable_scope("", reuse=True):
#         print(sess.run(tf.get_variable("v")))

# g=tf.Graph()
# g.device('/gpu:0')
# with g.as_default():
#     v = tf.get_variable("v", initializer=tf.ones_initializer()(shape=[2]))
# with tf.Session(graph=g) as sess:
#     tf.global_variables_initializer().run()
#     with tf.variable_scope("", reuse=True):
#         print(sess.run(tf.get_variable("v")))


# sess = tf.Session()
# print(sess.run(result))
# print(result.eval(session=sess))


# config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
# sess1 = tf.InteractiveSession(config=config)
# sess2 = tf.Session(config=config)


# #######################################

# weights = tf.Variable(tf.random_normal([2,3], stddev=2))

# weights = tf.Variable(tf.random_uniform([2,3]))

# weights = tf.Variable(tf.random_gamma([2,3], alpha=3))

# biases = tf.Variable(tf.zeros([3]))

# w2 = tf.Variable(weights.initialized_value())
# w3 = tf.Variable(weights.initialized_value() * 2.0)

###############3.4
##forward

##simple forward
# import tensorflow as tf
# w1=tf.Variable(tf.random_normal([2,3], stddev=1, seed=1))
# w2=tf.Variable(tf.random_normal([3,1], stddev=1, seed=1))

# x= tf.constant([[0.7, 0.9]])

# a=tf.matmul(x, w1)
# y=tf.matmul(a, w2)

# sess = tf.Session()
# # sess.run(w1.initializer)
# # sess.run(w2.initializer)

# init_op = tf.initializer_all_variables()
# sess.run(init_op)

# print(sess.run(y))
# sess.close()


##placeholder forward
# import tensorflow as tf
# w1 = tf.Variable(tf.random_normal([2,3], stddev=1))
# w2 = tf.Variable(tf.random_normal([3,1], stddev=1))

# x = tf.placeholder(tf.float32, shape=(3,2), name="input")
# a = tf.matmul(x, w1)
# y= tf.matmul(a, w2)

# sess = tf.Session()
# init_op = tf.global_variables_initializer()
# sess.run(init_op)

# print(sess.run(y, feed_dict={x:[[0.7, 0.9], [0.1, 0.4], [0.5, 0.8]]}))




# import tensorflow as tf

# from numpy.random import RandomState

# batch_size = 8

# w1 = tf.Variable(tf.random_normal([2,3], stddev=1, seed=1))
# w2 = tf.Variable(tf.random_normal([3,1], stddev=1, seed=1))

# x = tf.placeholder(tf.float32, shape=(None, 2), name='x-input')
# y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')

# a = tf.matmul(x, w1)
# y = tf.matmul(a, w2)

# cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1)))
# train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

# rdm = RandomState(1)
# dataset_size = 128
# X = rdm.rand(dataset_size, 2)
# Y = [[int(x1+x2 < 1)] for (x1, x2) in X]
# config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)

# with tf.Session(config=config) as sess:
#     init_op = tf.global_variables_initializer()
#     sess.run(init_op)
#     print(sess.run(w1))
#     print(sess.run(w2))
#     STEPS = 5000
#     for i in range(STEPS):
#         start = (i*batch_size) % dataset_size
#         end = min(start+batch_size, dataset_size)
#         sess.run(train_step, feed_dict = {x:X[start:end], y_:Y[start:end]})
#     print(sess.run(w1))
#     print(sess.run(w2))

#4.2
# import tensorflow as tf
# sess = tf.Session()
# y=tf.constant([1.0, 0.0, 0.0], name="y")
# y_ = tf.constant([0.5,0.4,0.1], name="y_")
# init_op = tf.global_variables_initializer()
# sess.run(init_op)
# print(tf.reduce_mean(y*tf.log(y_)).eval(session=sess))
# print(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_)).eval(session=sess))


# import tensorflow as tf
# from numpy.random import RandomState

# batch_size = 8
# x = tf.placeholder(tf.float32, shape=(None,2), name="x-input")
# y_ = tf.placeholder(tf.float32, shape=(None,1),name="y-input")
# w1=tf.Variable(tf.Variable(tf.random_normal([2,1], stddev=1, seed=1)))
# y=tf.matmul(x, w1)
# loss_less = 10
# loss_more = 1
# loss = tf.reduce_sum(tf.where(tf.greater(y, y_), (y-y_)*loss_more, (y_-y)*loss_less))
# train_step = tf.train.AdamOptimizer(0.001).minimize(loss)
# rdm = RandomState(1)
# dataset_size = 128
# X = rdm.rand(dataset_size, 2)
# Y = [[x1 + x2 + rdm.rand()/10.0-0.05] for(x1, x2) in X]
# with tf.Session() as sess:
#     init_op = tf.global_variables_initializer()
#     sess.run(init_op)
#     STEPS = 5000
#     for i in range(STEPS):
#         start = (i*batch_size)%dataset_size
#         end = min(start+batch_size, dataset_size)
#         sess.run(train_step, feed_dict={x:X[start:end], y_:Y[start:end]})
#         print(sess.run(w1))




 





























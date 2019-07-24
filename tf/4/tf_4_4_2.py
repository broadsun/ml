#coding=utf8
import tensorflow as tf

# w = tf.Variable(tf.random_normal([2,1], stddev=1, seed=1))
# y=tf.matmul(x, w)
# y=tf.constant(4)
# y_=tf.constant(8)
# loss = tf.reduce_mean(tf.square(y_ - y)) + tf.contrib.layers.l2_regularizer(lambda)(w)

weights = tf.constant([[1.0, -2.0],[-3.0, 4.0]])

with tf.Session() as sess:
    print(sess.run(tf.contrib.layers.l1_regularizer(0.5)(weights))) #5
    print(sess.run(tf.contrib.layers.l2_regularizer(0.5)(weights))) #7.5d


    def get_weight(shape, lambda):
        var = tf.Variable(tf.random_normal(shape), dtype=tf.float32)
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lambda)(var))
        return var
    x = tf.placeholder(tf.float32, shape=(None,2))
    y = tf.placeholder(tf.float32, shape=(None, 1))
    batch_size = 8
    layer_dimension = [2,10,10,10,1]
    n_layers=len(layer_dimension)
    cur_layer = x
    in_dimension = layer_dimension[0]
    for i in range(1, n_layers):
        out_dimension = layer_dimension[i]
        weight = get_weight([in_dimension, out_dimension], 0.001)
        bias = tf.Variable(tf.constant(0.1, shape=[out_dimension]))
        cur_layer = tf.nn.relu(tf.matmul(cur_layer, weight)+bias)
        in_dimension = out_dimension
    mse_loss = tf.reduce_mean(tf.square(y_ - cur_layer))
    tf.add_to_collection('losses', mse_loss)
    loss = tf.add_n(tf.get_collection('losses'))
    
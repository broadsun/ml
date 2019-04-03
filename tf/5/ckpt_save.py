import tensorflow as tf

v1 = tf.Variable(tf.random_normal([1], stddev=1, seed=1))
v2 = tf.Variable(tf.random_normal([1], stddev=1, seed=1))
result = v1 + v2
v3 = tf.Variable(tf.random_normal([1], stddev=1, seed=1))
init_op = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init_op)
    saver.save(sess, "save/model.ckpt")


with tf.Session() as sess:
    saver.restore(sess, "save/model.ckpt")
    print(sess.run(result))


saver = tf.train.import_meta_graph("save/model.ckpt.meta")

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    saver.restore(sess, "save/model.ckpt")
    print(sess.run(v1))
    print(sess.run(v2))
    print(sess.run(v3))
#coding=utf8
import tensorflow as tf

v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1")
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="v2")
result = v1+v2
result1 = v2+v2
#print(result.name)
init_op = tf.global_variables_initializer()

saver = tf.train.Saver()
saver.export_meta_graph("./model_json/model.ckpt.meta.json", as_text=True)
# with tf.Session() as sess:
#     sess.run(init_op)
#     saver.save(sess, "./model/model.ckpt")
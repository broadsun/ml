#coding=utf8
import tensorflow as tf

c = tf.constant("Hello distributed tensorflow")
server = tf.train.Server.create_local_server()
sess = tf.Session(server.target)
print(sess.run(c))
sess.close()
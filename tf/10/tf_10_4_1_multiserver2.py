#coding=utf8
import tensorflow as tf
print(help(tf.train.ClusterSpec));exit(0)
c = tf.constant("Hello from tensorflow server2")
cluster = tf.train.ClusterSpec(
    {"jobname":["localhost:2222", "localhost:2223"]})
server = tf.train.Server(cluster, job_name = "jobname", task_index = 1)
sess = tf.Session(server.target, config=tf.ConfigProto(log_device_placement=True))
print(sess.run(c))
sess.close()
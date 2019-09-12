#coding=utf8
import tensorflow as tf

c = tf.constant("Hello from tensorflow server1")
cluster = tf.train.ClusterSpec(
    {"jobname":["localhost:2222", "localhost:2223"]})
server = tf.train.Server(cluster, job_name = "jobname", task_index = 0)
sess = tf.Session(server.target, config=tf.ConfigProto(log_device_placement=True))
print(sess.run(c))
sess.close()
#coding=utf8
import tensorflow as tf
#会话表示方式1
sess = tf.Session()
print(sess.run(tf.constant([1,2])))
sess.close()

#会话表示方式2
with tf.Session() as sess:
    print(sess.run(tf.constant([1,2])))

#会话表示方式3
sess = tf.Session()
result = tf.constant([1,2]) + tf.constant([4,7])
with sess.as_default():
    print(result.eval())
print(result.eval(session=sess))
sess.close()

#会话表示方式4
sess=tf.InteractiveSession()
print(result.eval())
sess.close()

#可以通过ConfigProto来配置会话，比如并行的线程数，GPU分配策略，超时时间
config=tf.ConfigProto(allow_soft_placement=True,
                      log_device_placement=True)
sess1=tf.InteractiveSession(config=config)
sess2=tf.Session(config=config)
#为了保证程序的可移植性和灵活性，allow_soft_placement最好设置成True，
#log_device_placement设置成True方便调试


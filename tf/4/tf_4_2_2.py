#coding=utf8
import tensorflow as tf
from numpy.random import RandomState

batch_size = 8

x = tf.placeholder(tf.float32, shape=(None, 2), name='x-input')
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')

w1 = tf.Variable(tf.random_normal([2,1], stddev=1, seed=1))
y = tf.matmul(x, w1)

loss_less = 10
loss_more = 1
loss = tf.reduce_mean(tf.where(tf.greater(y,y_), (y-y_)*loss_more, (y_-y)*loss_less))
#定义损失函数为MSE
#loss = tf.losses.mean_squared_error(y, y_)
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

rdm = RandomState(1)

dataset_size = 128
X=rdm.rand(dataset_size, 2)
Y=[[x1+x2+rdm.rand()/10.0-0.05] for (x1, x2) in X]
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    STEPS = 5000
    for i in range(STEPS):
        start = (i*batch_size) % dataset_size
        end = min(start+batch_size, dataset_size)
        
        sess.run(train_step, feed_dict={x:X[start:end], y_:Y[start:end]})
        print(sess.run(w1))

'''
最终的结果是
[[1.019347 ]
 [1.0428089]]
定义损失函数使得预测少了的损失大，于是模型应该偏向多的方向预测。
重新定义损失函数，使得预测多了的损失大，于是模型应该偏向少的方向预测。

'''

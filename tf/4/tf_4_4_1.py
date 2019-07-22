#coding=utf8
import tensorflow as tf

#decayed_learning_rate = learning_rate * decay_rate^(global_step/decay_steps)
#上面这个式子的含义是每迭代decay_steps轮，乘以一次decay_rate。
#decay_steps是一次epoch需要的iteration数。 样本数/batchsize


#1. 学习率为1的时候，x在5和-5之间震荡。
def learning_rate_1():
    import tensorflow as tf
    TRAINING_STEPS = 10
    LEARNING_RATE = 1
    x = tf.Variable(tf.constant(5, dtype=tf.float32), name="x")
    y = tf.square(x)

    train_op = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(y)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(TRAINING_STEPS):
            sess.run(train_op)
            x_value = sess.run(x)
            print("After %s iteration(s): x%s is %f." % (i+1, i+1, x_value))
#2. 学习率为0.001的时候，下降速度过慢，在901轮时才收敛到0.823355
def learning_rate_0_001():
    import tensorflow as tf
    TRAINING_STEPS = 1000
    LEARNING_RATE = 0.001
    x = tf.Variable(tf.constant(5, dtype=tf.float32), name="x")
    y = tf.square(x)

    train_op = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(y)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(TRAINING_STEPS):
            sess.run(train_op)
            x_value = sess.run(x)
            print("After %s iteration(s): x%s is %f." % (i+1, i+1, x_value))


def decay_learning_rate():   
    TRAINING_STEPSTRAININ  = 100
    global_step = tf.Variable(0)
    LEARNING_RATE = tf.train.exponential_decay()

    x = tf.Variable(tf.constant(5, dtype=tf.float32), name="x")
    y = tf.square(x)
    train_op = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(
        y, global_step=global_step)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(TRAINING_STEPS):
            sess.run(train_op)
            if i % 10 == 0:
                LEARNING_RATE_value = sess.run(LEARNING_RATE)
                x_value = sess.run(x)
                print "After %s iteration(s): x%s is %f, learning rate is %f."% (i+1, i+1, x_value, LEARNING_RATE_value)
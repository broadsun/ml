# #coding=utf8
# import tensorflow.contrib.slim as slim
# with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride = 1, padding ='SAME'):
#         net = np.array([1,2,3])
#         with tf.variable_scope("Mixed_7c"):
#             with tf.variable_scope('Branch_0'):
#                 branch_0 = slim.conv2d(net, 320, [1,1], scope='Conv2d_0a_1x1')
#             with tf.variable_scope('Branch_1'):
#                 branch_1 = slim.conv2d([net, 384, [1,1]], scope = 'Conv2d_0a_1x1')
#                 branch_1 = tf.concat(3, [
#                     slim.conv2d(branch_1, 384, [1,3], scope='Conv2d_0b_1x3')
#                 ])

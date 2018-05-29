# coding=utf-8
# 单机多卡：
# 一般采用共享操作定义在cpu上，然后并行操作定义在各自的gpu上，比如对于深度学习来说，我们一把把参数定义、参数梯度更新统一放在cpu上
# 各个gpu通过各自计算各自batch 数据的梯度值，然后统一传到cpu上，由cpu计算求取平均值，cpu更新参数。
# 具体的深度学习多卡训练代码，请参考：https://github.com/tensorflow/models/blob/master/inception/inception/inception_train.py
import tensorflow as tf

with tf.device('/cpu:0'):
    w = tf.get_variable('w', (2, 2), tf.float32, initializer=tf.constant_initializer(2))
    b = tf.get_variable('b', (2, 2), tf.float32, initializer=tf.constant_initializer(5))
    addwb = w + b
# with tf.device('/gpu:0'):
#     addwb = w + b
with tf.device('/gpu:1'):
    mutwb = w * b

ini = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(ini)
    while 1:
        print(sess.run([addwb, mutwb]) )
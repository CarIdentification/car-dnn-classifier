#  coding=utf-8
# https://blog.csdn.net/hjimce/article/details/61197190
# https://www.tensorflow.org/versions/master/get_started/
# 单机单卡
# 对于单机单卡，可以把参数和计算都定义再gpu上，不过如果参数模型比较大，显存不足等情况，就得放在cpu上
import tensorflow as tf

with tf.device('/cpu:0'):  # 也可以放在gpu上
    w = tf.get_variable('w', (2, 2), tf.float32, initializer=tf.constant_initializer(2))
    b = tf.get_variable('b', (2, 2), tf.float32, initializer=tf.constant_initializer(5))
    addwb = w + b
    mutwb = w * b

# with tf.device('/gpu:0'):
#     addwb = w + b
#     mutwb = w * b

ini = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(ini)
    np1, np2 = sess.run([addwb, mutwb])
    print(np1)
    print(np2)
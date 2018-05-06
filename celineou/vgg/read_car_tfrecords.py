import os;
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

cwd='/Users/celineou/Desktop/Study/毕业设计/car_data/'

##读取TFRECORD文件

filename = cwd + 'car_train.tfrecords'
def read_and_decode(filename): # 读入car_train.tfrecords
    # filename_queue = tf.train.string_input_producer([filename])#生成一个queue队列
    #
    # reader = tf.TFRecordReader()
    # _, serialized_example = reader.read(filename_queue)#返回文件名和文件
    # features = tf.parse_single_example(serialized_example,
    #                                    features={
    #                                        'label': tf.FixedLenFeature([], tf.int64),
    #                                        'img_raw' : tf.FixedLenFeature([], tf.string),
    #                                    })#将image数据和label取出来
    #
    # img = tf.decode_raw(features['img_raw'], tf.uint8)
    # img = tf.reshape(img, [128, 128, 3])  #reshape为128*128的3通道图片
    # img = tf.cast(img, tf.float32) * (1. / 255) - 0.5 #在流中抛出img张量

    filename_queue = tf.train.string_input_producer([filename]) #读入流中
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)   #返回文件名和文件
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw' : tf.FixedLenFeature([], tf.string),
                                       })  #取出包含image和label的feature对象
    image = tf.decode_raw(features['img_raw'], tf.uint8)
    image = tf.reshape(image, [224, 224, 3])
    label = tf.cast(features['label'], tf.int32)
    with tf.Session() as sess: #开始一个会话
        init_op = tf.initialize_all_variables()
        sess.run(init_op)
        coord=tf.train.Coordinator()
        threads= tf.train.start_queue_runners(coord=coord)
        for i in range(18):
            example, l = sess.run([image,label])#在会话中取出image和label
            img=Image.fromarray(example, 'RGB')#这里Image是之前提到的
            img.save(cwd+str(i)+'_''Label_'+str(l)+'.jpg')#存下图片
            print(example, l)
        coord.request_stop()
        coord.join(threads)



print(read_and_decode(filename))
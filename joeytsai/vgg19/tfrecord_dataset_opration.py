import cv2

#https://www.cnblogs.com/wktwj/p/7257526.html
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import tensorflow as tf
import numpy as np
from PIL import Image

from joeytsai.vgg19.test_variables import test_variables

file_path = "./train_carimg/classes/"
output_path  = "./train_carimg/classes/tfrecords/"
# 转化为tfRecord
def trans2tfRecord(file_path,name,output_path):
    class_list = {"1", "2", "3", "4"}
    if not os.path.exists(output_path) or os.path.isfile(output_path):
        os.makedirs(output_path)
    output_file = output_path + name + '.tfrecords'
    writer = tf.python_io.TFRecordWriter(output_file)
    for index , type_name in enumerate(class_list):
        type_dir = file_path + type_name
        for file_name in os.listdir(type_dir):
            file_path_temp = type_dir + "/" + file_name
            img = Image.open(file_path_temp)
            img = img.resize((224, 224))
            image_raw= img.tobytes()
            example = tf.train.Example(features=tf.train.Features(feature={
                    'img_byte':_bytes_feature(image_raw),
                     'label': _int64_feature(index)
                    }))
            writer.write(example.SerializeToString())
    writer.close()

def read_and_decode(filename): # read iris_contact.tfrecords
    filename_queue = tf.train.string_input_producer([filename])# create a queue

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)#return file_name and file
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_byte' : tf.FixedLenFeature([], tf.string),
                                       })#return image and label

    image = tf.decode_raw(features['img_byte'], tf.uint8)
    image = tf.reshape(image, [224, 224, 3])  #reshape image to 512*80*3

    #归一化 nomalize
    image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
    #标准化 standardization
    image = tf.image.per_image_standardization(image)

    label = tf.cast(features['label'], tf.int32) #throw label tensor

    test_variables("label",label)
    return image, label

def tfrecord2pic(input_path,filename,output_path):
    filename_queue = tf.train.string_input_producer([input_path+filename])
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)  # return file and file_name
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_byte': tf.FixedLenFeature([], tf.string),
                                       })
    image = tf.decode_raw(features['img_byte'], tf.uint8)
    image = tf.reshape(image, [224, 224, 3])
    label = tf.cast(features['label'], tf.int32)
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()


        sess.run(init_op)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        for i in range(30):
            example, l = sess.run([image, label])  # take out image and label

            img = Image.fromarray(example, 'RGB')
            if not os.path.exists(output_path) or os.path.isfile(output_path):
                os.makedirs(output_path)
            img.save(output_path + str(i) + '_''Label_' + str(l) + '.jpg')  # save image
            print(example, l)
        coord.request_stop()
        coord.join(threads)

def _int64_feature(value):
     return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
     return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))




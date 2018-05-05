import os;
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

cwd='/Users/celineou/Desktop/Study/毕业设计/car_data/'
classes = {'1','2','3','4'}  #正面 斜侧面 侧面 正背面
writer = tf.python_io.TFRecordWriter("car_train.tfrecords")  #要生成的文件

for index,name in enumerate(classes):
    class_path = cwd + '/'
    for img_name in os.listdir(class_path):
        img_path = class_path + img_name #每张图片的地址

        img = Image.open(img_path)
        img = img.resize((128,128))
        img_raw = img.tobytes() #将图片转化为二进制格式
        example = tf.train.Example(features = tf.train.Feature(feature={
            "label":tf.train.Feature(int64_list = tf.train.Int64List(value=[index])),
            "img_raw":tf.train.Feature(bytes_list = tf.train.BytesList(value=[img_raw]))
        })) #example对象对lable和image的数据进行封装
        writer.write(example.SerializeToString()) #序列化为字符串
writer.close()

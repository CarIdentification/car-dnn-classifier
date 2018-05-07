import os
import tensorflow as tf
from PIL import Image  #注意Image,后面会用到
import matplotlib.pyplot as plt
import numpy as np

filePath = "/Users/cai/personal/car/tfSource/car-dnn-classifier/cjt/traindataset/" #图片路径
classes = {"1","2","3","4"} #1正面 2侧面 3后面 4斜侧面
writerFile = tf.python_io.TFRecordWriter("train_test.tfrecords") # 要输出的文件名字
sess = tf.Session()

for index, name in enumerate(classes):
    class_path = filePath + name + '/' # 四个面的文件夹路径
    for img_name in os.listdir(class_path): #list出所有图片
        if img_name.startswith('.'): # mac制作数据时候忽略掉隐藏文件
            continue

        print(img_name.startswith('.'))

        img_path = class_path + img_name # 每一个图片的路径

        img = Image.open(img_path) #打开文件流
        img_load = sess.run(tf.reshape(img,[224,224,3])) #直接计算结果返回numpy数组
        print(img_load)
        img_bytes = img_load.tobytes() # 将张量转化为字节类型.
        # img = img.resize((224,224)) #对图片进行缩放大小，进行图像处理
        example = tf.train.Example(features=tf.train.Features(feature={
            "label" : tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
            'img_raw' : tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_bytes]))
        })) #example对象将label和image进行封装
        writerFile.write(example.SerializeToString()) #序列化后写入文件中

writerFile.close()
sess.close()

import tensorflow as tf
import os
from PIL import Image

base_path = "G:\\Projects\\PyCharmProjects\\car-dnn-classifier\\hzd\\vgg\\"
class_list = {"1", "2", "3", "4"}
writer = tf.python_io.TFRecordWriter("car.tfrecords")

for index, type_name in enumerate(class_list):
    type_dir = base_path + type_name + "\\"
    for file_name in os.listdir(type_dir):
        file_path = type_dir + file_name
        img = Image.open(file_path)

        img_byte = img.tobytes()
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "label": tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[index])
                    ),
                    'img_byte': tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[img_byte])
                    )
                }
            )
        )

        writer.write(example.SerializeToString())
writer.close()

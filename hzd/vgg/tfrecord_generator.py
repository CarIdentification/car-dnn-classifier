import tensorflow as tf
import os
from PIL import Image

base_path = "G:\\Projects\\PyCharmProjects\\car-dnn-classifier\\hzd\\vgg\\stanford_car_dataset_solved\\train\\"
class_list = {"1", "2", "3", "4"}
writer = tf.python_io.TFRecordWriter("car.tfrecords")
sess = tf.Session()
num = 0
for i in range(1, 197):
    type_dir = base_path + str(i) + "\\"
    for file_name in os.listdir(type_dir):
        num = num + 1
        file_path = type_dir + file_name
        image_raw_data = tf.gfile.FastGFile(file_path, 'rb').read()
        image = tf.image.decode_jpeg(image_raw_data)
        image = sess.run(tf.reshape(image, [224, 224, 3]))
        print(image)
        image = image.tobytes()
        example = tf.train.Example(
            features=tf.train.Features(
                feature={
                    "label": tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[i-1])
                    ),
                    'img_byte': tf.train.Feature(
                        bytes_list=tf.train.BytesList(value=[image])
                    )
                }
            )
        )
        writer.write(example.SerializeToString())

writer.close()
print("num : ", num)

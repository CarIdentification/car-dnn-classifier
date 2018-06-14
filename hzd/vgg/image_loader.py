import os
from PIL import Image
import numpy as np
import random
import keras


class ImageLoader:
    x_train = None
    y_label = None
    data_num = 0
    batch_size = 32
    index = 0

    @staticmethod
    def re_shuffle():
        permutation = np.random.permutation(ImageLoader.x_train.shape[0])
        ImageLoader.x_train = ImageLoader.x_train[permutation]
        ImageLoader.y_label = ImageLoader.y_label[permutation]

    def __init__(self, base_path, batch_size=32, shuffle=True):
        num = 0
        x_train = []
        y_label = []
        ImageLoader.batch_size = batch_size
        for i in range(1, 197):
            type_dir = base_path + str(i) + "\\"
            for file_name in os.listdir(type_dir):
                num = num + 1
                file_path = type_dir + file_name
                pic_arr = np.array(Image.open(file_path))
                x_train.append(pic_arr)
                y_label.append(i - 1)
        print(num, " pics was loaded")
        ImageLoader.data_num = num
        if shuffle:
            zipped_data = list(zip(x_train, y_label))
            random.shuffle(zipped_data)
            x_train[:], y_label[:] = zip(*zipped_data)
            print("shuffle done")
        ImageLoader.x_train = np.array(x_train)
        ImageLoader.y_label = np.array(y_label)

    @staticmethod
    def generate():
        idx = ImageLoader.index
        idx_tail = ImageLoader.index + ImageLoader.batch_size
        while True:
            if idx_tail < ImageLoader.data_num:
                yield ImageLoader.x_train[idx:idx_tail], keras.utils.to_categorical(ImageLoader.y_label[idx:idx_tail],
                                                          196)
                ImageLoader.index = idx_tail
            else:
                idx_tail = idx_tail - ImageLoader.data_num
                yield ImageLoader.x_train[idx:] + ImageLoader.x_train[:idx_tail], keras.utils.to_categorical(
                    ImageLoader.y_label[idx:] + ImageLoader.y_label[:idx_tail], 196)
                ImageLoader.re_shuffle()
                ImageLoader.index = 0

# loader = ImageLoader(batch_size=1,base_path="G:\\Projects\\PyCharmProjects\\car-dnn-classifier\\hzd\\vgg\\stanford_car_dataset_solved\\train\\")
# a = loader.generate()
# print(a)

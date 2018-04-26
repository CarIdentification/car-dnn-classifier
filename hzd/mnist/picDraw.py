import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf

# 用于打印mnist数据集

mnist = tf.contrib.learn.datasets.load_dataset("mnist")
train_data = mnist.train.images  # Returns np.array
train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
picCount = 0
# for index, pic_data in enumerate(train_data):
#     picCount = picCount + 1
#     if picCount == 20:
#         break
#     pic_data = np.reshape(pic_data, (28, 28))
#     print(train_labels[index], " ")
#     plt.imshow(pic_data, cmap=cm.binary)
#     plt.show()
for index, pic_label in enumerate(train_labels):
    if pic_label == 9:
        pic_data = np.reshape(train_data[index], (28, 28))
        plt.imshow(pic_data, cmap=cm.binary)
        plt.show()

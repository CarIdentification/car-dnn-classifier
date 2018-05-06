from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

class Vgg19(object):
    """
    VGG19
    """

    ###初始化
    def __init__(self, bgr_image, num_class, vgg19_npy_path=None, trainable=True, dropout=0.5):
        if vgg19_npy_path is not None:
            self.data_dict = np.load(vgg19_npy_path, encoding='latin1').item()
        else:
            self.data_dict = None
        self.BGR_IMAGE = bgr_image
        self.NUM_CLASS = num_class
        self.var_dict = {}
        self.trainable = trainable
        self.dropout = dropout

        self.build()

    ###开始搭建模型
    def build(self, train_mode=None):

        ##input (224*224 RGB image)  conv3-64*2
        self.conv1_1 = self.conv_layer(self.BGR_IMAGE, 3, 64, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, 64, 64, "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        ##maxpool conv3-128*2
        self.conv2_1 = self.conv_layer(self.pool1, 64, 128, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, 128, 128, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        ##maxpool conv3-256*4
        self.conv3_1 = self.conv_layer(self.pool2, 128, 256, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, 256, 256, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, 256, 256, "conv3_3")
        self.conv3_4 = self.conv_layer(self.conv3_3, 256, 256, "conv3_4")
        self.pool3 = self.max_pool(self.conv3_4, 'pool3')

        ##maxpool conv3-512*4
        self.conv4_1 = self.conv_layer(self.pool3, 256, 512, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, 512, 512, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, 512, 512, "conv4_3")
        self.conv4_4 = self.conv_layer(self.conv4_3, 512, 512, "conv4_4")
        self.pool4 = self.max_pool(self.conv4_4, 'pool4')

        ##maxpool conv3-512*4
        self.conv5_1 = self.conv_layer(self.pool4, 512, 512, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, 512, 512, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, 512, 512, "conv5_3")
        self.conv5_4 = self.conv_layer(self.conv5_3, 512, 512, "conv5_4")
        self.pool5 = self.max_pool(self.conv5_4, 'pool5')

        ##全连接层
        ##FC-4096
        self.fc6 = self.fc_layer(self.pool5, 25088, 4096, "fc6")
        self.relu6 = tf.nn.relu(self.fc6)
        if train_mode is not None:
            self.relu6 = tf.cond(train_mode, lambda: tf.nn.dropout(self.relu6, self.dropout), lambda: self.relu6)
        elif train_mode:
            self.relu6 = tf.nn.dropout(self.relu6, self.dropout)

        ##FC-4096
        self.fc7 = self.fc_layer(self.relu6, 4096, 4096, "fc7")
        self.relu7 = tf.nn.relu(self.fc7)
        if train_mode is not None:
            self.relu7 = tf.cond(train_mode, lambda: tf.nn.dropout(self.relu7, self.dropout), lambda: self.relu7)
        elif train_mode:
            self.relu7 = tf.nn.dropout(self.relu7, self.dropout)

        self.fc8 = self.fc_layer(self.relu7, 4096, self.NUM_CLASS, "fc8")

        self.prob = tf.nn.softmax(self.fc8, name="prob")

        self.data_dict = None
# coding: UTF-8
import tensorflow as tf
import numpy as np
#https://blog.csdn.net/u014365862/article/details/77833481
# https://blog.csdn.net/cxmscb/article/details/71023576
# 卷积层计算 ： https://blog.csdn.net/zhangwei15hh/article/details/78417789
# reshape操作 ： https://blog.csdn.net/lxg0807/article/details/53021859
# conv2d参数详解 ： https://blog.csdn.net/lujiandong1/article/details/53728053
# padding操作计算 ： https://blog.csdn.net/wkk15903468980/article/details/75024467
# dropout : https://blog.csdn.net/williamyi96/article/details/77544536
###########################################################################################
# 1. conv3 - 64（卷积核的数量）：kernel size:3 stride:1 pad:1
# 像素：（224-3+2*1）/1+1=224 224*224*64
# 参数： （3*3*3）*64 =1728
# 2. conv3 - 64：kernel size:3 stride:1 pad:1
# 像素： （224-3+1*2）/1+1=224 224*224*64
# 参数： （3*3*64）*64 =36864
# 3. pool2 kernel size:2 stride:2 pad:0
# 像素： （224-2）/2 = 112 112*112*64
# 参数： 0
# 4.conv3-128:kernel size:3 stride:1 pad:1
# 像素： （112-3+2*1）/1+1 = 112 112*112*128
# 参数： （3*3*64）*128 =73728
# 5.conv3-128:kernel size:3 stride:1 pad:1
# 像素： （112-3+2*1）/1+1 = 112 112*112*128
# 参数： （3*3*128）*128 =147456
# 6.pool2: kernel size:2 stride:2 pad:0
# 像素： （112-2）/2+1=56 56*56*128
# 参数：0
# 7.conv3-256: kernel size:3 stride:1 pad:1
# 像素： （56-3+2*1）/1+1=56 56*56*256
# 参数：（3*3*128）*256=294912
# 8.conv3-256: kernel size:3 stride:1 pad:1
# 像素： （56-3+2*1）/1+1=56 56*56*256
# 参数：（3*3*256）*256=589824
# 9.conv3-256: kernel size:3 stride:1 pad:1
# 像素： （56-3+2*1）/1+1=56 56*56*256
# 参数：（3*3*256）*256=589824
# 10.pool2: kernel size:2 stride:2 pad:0
# 像素：（56 - 2）/2+1=28 28*28*256
# 参数：0
# 11. conv3-512:kernel size:3 stride:1 pad:1
# 像素：（28-3+2*1）/1+1=28 28*28*512
# 参数：（3*3*256）*512 = 1179648
# 12. conv3-512:kernel size:3 stride:1 pad:1
# 像素：（28-3+2*1）/1+1=28 28*28*512
# 参数：（3*3*512）*512 = 2359296
# 13. conv3-512:kernel size:3 stride:1 pad:1
# 像素：（28-3+2*1）/1+1=28 28*28*512
# 参数：（3*3*512）*512 = 2359296
# 14.pool2: kernel size:2 stride:2 pad:0
# 像素：（28-2）/2+1=14 14*14*512
# 参数： 0
# 15. conv3-512:kernel size:3 stride:1 pad:1
# 像素：（14-3+2*1）/1+1=14 14*14*512
# 参数：（3*3*512）*512 = 2359296
# 16. conv3-512:kernel size:3 stride:1 pad:1
# 像素：（14-3+2*1）/1+1=14 14*14*512
# 参数：（3*3*512）*512 = 2359296
# 17. conv3-512:kernel size:3 stride:1 pad:1
# 像素：（14-3+2*1）/1+1=14 14*14*512
# 参数：（3*3*512）*512 = 2359296
# 18.pool2:kernel size:2 stride:2 pad:0
# 像素：（14-2）/2+1=7 7*7*512
# 参数：0

# 19.FC: 4096 neurons
# 像素：1*1*4096
# 参数：7*7*512*4096 = 102760448

# 20.FC: 4096 neurons
# 像素：1*1*4096
# 参数：4096*4096 = 16777216
# 21.FC：1000 neurons
# 像素：1*1*1000
# 参数：4096*1000=4096000
###########################################################################################
# define different layer functions
# we usually don't do convolution and pooling on batch and channel
def maxPoolLayer(x, kHeight, kWidth, strideX, strideY, name, padding = "SAME"):
    """max-pooling"""
    # tf.nn.max_pool参数解释 ： https://blog.csdn.net/mao_xiao_feng/article/details/53453926
    return tf.nn.max_pool(x, ksize = [1, kHeight, kWidth, 1],
                          strides = [1, strideX, strideY, 1], padding = padding, name = name)

def dropout(x, keepPro, name = None):
    """dropout"""
    #keepPro训练节点保留率
    return tf.nn.dropout(x, keepPro, name)

def fcLayer(x, inputD, outputD, reluFlag, name):
    """fully-connect"""
    with tf.variable_scope(name) as scope:
        #get_variable : https://blog.csdn.net/u013713117/article/details/66001439
        w = tf.get_variable("w", shape = [inputD, outputD], dtype = "float")
        b = tf.get_variable("b", [outputD], dtype = "float")
        #tf.nn.xw_plus_b() : https://blog.csdn.net/qq_31486503/article/details/77155526
        #tf.nn.xw_plus_b() => matmul(x, weights) + biases. :http://www.360doc.com/content/17/0321/10/10408243_638692495.shtml
        out = tf.nn.xw_plus_b(x, w, b, name = scope.name)
        if reluFlag:
            return tf.nn.relu(out)
        else:
            return out

def convLayer(x, kHeight, kWidth, strideX, strideY,
              featureNum, name, padding = 'SAME'):
    """convlutional"""
    # conv2d参数详解 ： https://blog.csdn.net/lujiandong1/article/details/53728053
    channel = int(x.get_shape()[-1])
    with tf.variable_scope(name) as scope:
        w = tf.get_variable("w", shape = [kHeight, kWidth, channel, featureNum])
        b = tf.get_variable("b", shape = [featureNum])
        featureMap = tf.nn.conv2d(x, w, strides = [1, strideY, strideX, 1], padding = padding)
        # https://www.cnblogs.com/lovephysics/p/7222022.html
        out = tf.nn.bias_add(featureMap, b)
        return tf.nn.relu(tf.reshape(out, featureMap.get_shape().as_list()), name = scope.name)

def loss(logits,label_batches):
     cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,labels=label_batches)
     cost = tf.reduce_mean(cross_entropy)
     return cost

def training(loss,lr):
     train_op = tf.train.RMSPropOptimizer(lr,0.9).minimize(loss)
     return train_op

def get_accuracy(logits,labels):
     acc = tf.nn.in_top_k(logits,labels,1)
     acc = tf.cast(acc,tf.float32)
     acc = tf.reduce_mean(acc)
     return acc

def vgg19_model(x, keepPro, classNum, skip):
    """build model"""
    conv1_1 = convLayer(x, 3, 3, 1, 1, 64, "conv1_1" )
    conv1_2 = convLayer(conv1_1, 3, 3, 1, 1, 64, "conv1_2")
    pool1 = maxPoolLayer(conv1_2, 2, 2, 2, 2, "pool1")

    conv2_1 = convLayer(pool1, 3, 3, 1, 1, 128, "conv2_1")
    conv2_2 = convLayer(conv2_1, 3, 3, 1, 1, 128, "conv2_2")
    pool2 = maxPoolLayer(conv2_2, 2, 2, 2, 2, "pool2")

    conv3_1 = convLayer(pool2, 3, 3, 1, 1, 256, "conv3_1")
    conv3_2 = convLayer(conv3_1, 3, 3, 1, 1, 256, "conv3_2")
    conv3_3 = convLayer(conv3_2, 3, 3, 1, 1, 256, "conv3_3")
    conv3_4 = convLayer(conv3_3, 3, 3, 1, 1, 256, "conv3_4")
    pool3 = maxPoolLayer(conv3_4, 2, 2, 2, 2, "pool3")

    conv4_1 = convLayer(pool3, 3, 3, 1, 1, 512, "conv4_1")
    conv4_2 = convLayer(conv4_1, 3, 3, 1, 1, 512, "conv4_2")
    conv4_3 = convLayer(conv4_2, 3, 3, 1, 1, 512, "conv4_3")
    conv4_4 = convLayer(conv4_3, 3, 3, 1, 1, 512, "conv4_4")
    pool4 = maxPoolLayer(conv4_4, 2, 2, 2, 2, "pool4")

    conv5_1 = convLayer(pool4, 3, 3, 1, 1, 512, "conv5_1")
    conv5_2 = convLayer(conv5_1, 3, 3, 1, 1, 512, "conv5_2")
    conv5_3 = convLayer(conv5_2, 3, 3, 1, 1, 512, "conv5_3")
    conv5_4 = convLayer(conv5_3, 3, 3, 1, 1, 512, "conv5_4")
    pool5 = maxPoolLayer(conv5_4, 2, 2, 2, 2, "pool5")

    fcIn = tf.reshape(pool5, [-1, 7*7*512])
    fc6 = fcLayer(fcIn, 7 * 7 * 512 , 4096, True, "fc6")
    dropout1 = dropout(fc6, keepPro)

    fc7 = fcLayer(dropout1, 4096, 4096, True, "fc7")
    dropout2 = dropout(fc7, keepPro)
    #softmax
    fc8 = fcLayer(dropout2, 4096, classNum, True, "fc8")
    return fc8


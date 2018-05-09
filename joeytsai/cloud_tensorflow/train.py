
import tensorflow as tf
import numpy as np


import tensorflow as tf
def test_variables(variable_name,variable):
    sess = tf.Session()
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    l = sess.run(variable)
    coord.request_stop()
    coord.join(threads)
    sess.close()
    print(" ########################  begin  {}  #########################".format(variable_name) ,"\n",l, "\n","#########################   end  {}   #########################".format(variable_name))

def read_and_decode(filename): # read iris_contact.tfrecords
    filename_queue = tf.train.string_input_producer([filename])# create a queue

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)#return file_name and file
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_byte' : tf.FixedLenFeature([], tf.string),
                                       })#return image and label

    img = tf.decode_raw(features['img_byte'], tf.uint8)
    img = tf.reshape(img, [224, 224, 3])  #reshape image to 512*80*3
    img = tf.cast(img, tf.float32) * (1. / 255) - 0.5 #throw img tensor
    image = tf.image.per_image_standardization(img)
    label = tf.cast(features['label'], tf.int32) #throw label tensor
    return img, label


def _int64_feature(value):
     return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
     return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))



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



#http://www.cnblogs.com/wktwj/p/7227544.html
if __name__ == '__main__':

   (image , label) = read_and_decode("./car.tfrecords")
   # (image , label) = read_and_decode("drive/cartensorflow/car/car.tfrecords")
   #http://www.cnblogs.com/wktwj/p/7227544.html

   image_batches, label_batches = tf.train.shuffle_batch([image, label], batch_size=50, capacity=100,min_after_dequeue=75)

   p = vgg19_model(image_batches,0.5,4,"没有用的参数")
   cost = loss(p, label_batches)
   train_op = training(cost, 0.001)

   acc = get_accuracy(p, label_batches)

   sess = tf.Session()
   init = tf.global_variables_initializer()
   sess.run(init)

   coord = tf.train.Coordinator()
   threads = tf.train.start_queue_runners(sess=sess, coord=coord)

   try:
       for step in np.arange(20000):
           print(step)
           if coord.should_stop():
               break
           a  = 1
           _, train_acc, train_loss  = sess.run([train_op, acc, cost ])
           print("loss:{} accuracy:{}".format(train_loss, train_acc))
           test_variables("p", p)
   except tf.errors.OutOfRangeError:
       print("Done!!!")
   finally:
       coord.request_stop()
   coord.join(threads)
   sess.close()
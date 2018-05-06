from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import cjt.tfrecords_input_fn as tr_input_fn

tf.logging.set_verbosity(tf.logging.INFO)  # 输出训练中的损失信息


def tfrecords_input_fn(file_paths, batch_size=100, num_epochs=None, shuffle=False):
    def data_generate():
        dataset = tf.data.TFRecordDataset(file_paths)
        dataset = dataset.map(parser)
        dataset = dataset.shuffle(buffer_size=10000)  # 在训练的时候一般需要将输入数据进行顺序打乱提高训练的泛化性
        dataset = dataset.batch(32)  # 单次读取的batch大小
        dataset = dataset.repeat(num_epochs)  # 数据集的重复使用次数，为空的话则无线循环
        iterator = dataset.make_one_shot_iterator()

        features, labels = iterator.get_next()
        return {"image_raw": features}, labels

    def parser(record):
        keys_map = {
            "label": tf.FixedLenFeature((), tf.int64, tf.zeros([], dtype=tf.int64)),
            "image_raw": tf.FixedLenFeature((), tf.string, default_value="")
        }
        parsed_data = tf.parse_single_example(record, keys_map)
        img = tf.decode_raw(parsed_data["image_raw"],tf.float32)
        img = tf.cast(img, tf.float32)
        label = parsed_data["label"]

        return img, label

    return data_generate

def generate_conv(inputs, filters):
    return tf.layers.conv2d(
        inputs=inputs,  # 使用的输入层
        filters=filters,
        kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
        kernel_size=3,  # 过滤器大小
        padding="same",  # 不维持输入输出层相同大小
        activation=tf.nn.relu)  # 激活函数使用Relu


def generate_max_pooling(inputs):
    return tf.layers.max_pooling2d(inputs=inputs, pool_size=[2, 2], strides=2)


def vgg_model_fn(features, labels, mode):
    # Input Layer
    input_layer = tf.reshape(features["img_raw"], [-1, 224, 224, 3 ])

    # 加权层1
    conv1 = generate_conv(input_layer, 64)  # 224,224,64
    conv2 = generate_conv(conv1, 64)  # 224,224,64

    pool1 = generate_max_pooling(conv2)  # 112,112,64

    # 加权层2
    conv3 = generate_conv(pool1, 128)  # 112,112,128
    conv4 = generate_conv(conv3, 128)  # 112,112,128

    pool2 = generate_max_pooling(conv4)  # 56,56,128

    # 加权层3
    conv5 = generate_conv(pool2, 256)  # 56,56,256
    conv6 = generate_conv(conv5, 256)  # 56,56,256
    conv7 = generate_conv(conv6, 256)  # 56,56,256
    conv8 = generate_conv(conv7, 256)  # 56,56,256

    pool3 = generate_max_pooling(conv8)  # 28,28,256

    # 加权层4
    conv9 = generate_conv(pool3, 512)  # 28,28,512
    conv10 = generate_conv(conv9, 512)  # 28,28,512
    conv11 = generate_conv(conv10, 512)  # 28,28,512
    conv12 = generate_conv(conv11, 512)  # 28,28,512

    pool4 = generate_max_pooling(conv12)  # 14,14,512

    # 加权层5
    conv13 = generate_conv(pool4, 512)  # 14,14,512
    conv14 = generate_conv(conv13, 512)  # 14,14,512
    conv15 = generate_conv(conv14, 512)  # 14,14,512
    conv16 = generate_conv(conv15, 512)  # 14,14,512

    pool5 = generate_max_pooling(conv16)  # 7,7,512

    # Dense Layer 全连接层
    pool5_flat = tf.reshape(pool5, [-1, 7 * 7 * 512])  # 矩阵转换为二维[n,7*7*512]
    dense1 = tf.layers.dense(inputs=pool5_flat, units=4096, activation=tf.nn.relu)  # 进行4096单元的全连接，激活函数为RELU
    dropout1 = tf.layers.dropout(
        inputs=dense1, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)  # 插入一个dropout层 丢弃50%节点

    dense2 = tf.layers.dense(inputs=dropout1, units=4096, activation=tf.nn.relu)  # 进行4096单元的全连接，激活函数为RELU
    dropout1 = tf.layers.dropout(
        inputs=dense2, rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)  # 插入一个dropout层 丢弃50%节点

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout1, units=4)  # 输出层  输出类型为四个方向   1:前方 2:侧面 3:前面+侧面 4:后面

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),  # 返回类别id
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")  # 将输出对数转换成概率
    }

    if mode == tf.estimator.ModeKeys.PREDICT:  # 模型在预测模式下返回的结果
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)   计算损失值   用于训练及评估模式
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)
    learning_rate = tf.train.exponential_decay(0.01, tf.train.get_global_step(), 10, 0.0005)  # 计算出按步数递减的学习率

    if mode == tf.estimator.ModeKeys.TRAIN:  # 如果是训练模式
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)  # 创建一个学习率0.01
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)   如果是评估模式  需要返回各项指标供研究者评估
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}  # 这里返回的只有正确率
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(argv):
    vgg_car_classifier = tf.estimator.Estimator(
        model_fn=vgg_model_fn, model_dir="/tmp/mnist_convnet_model")  # 新建自定义Estimator 传入自定义的模型函数   选定模型存放的目录
    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)  # 每训练50步输出该次结果probabilities的softmax_tensor层
    # Train the model
    train_input_fn = tr_input_fn.tfRecord_input_fn("train_test.tfrecords",1,None)
    # vgg_car_classifier.train(  # 训练
    #     input_fn=train_input_fn,
    #     steps=20000,
    #     hooks=[logging_hook])
    eval_input_fn =  tr_input_fn.tfRecord_input_fn("train_test.tfrecords",1,None)# 评估师如函数

    eval_results = vgg_car_classifier.evaluate(input_fn=eval_input_fn)  # 评估当前模型
    print(eval_results)

if __name__ == "__main__":
    tf.app.run()

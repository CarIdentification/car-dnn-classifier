from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from PIL import Image

# Imports
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import tensorflow as tf


def get_predict_data():
    file_set = ["predict/6.png", "predict/8.png", "predict/9.jpg"]
    result = []
    for file_name in file_set:
        img = Image.open(file_name)
        img = img.convert("I")
        data = img.getdata()
        data = 1 - (np.matrix(data, dtype='float32') / 255)  # 数据转换为矩阵并/255转换为灰度矩阵 并取反
        data[data < 0.4] = 0  # 去除背景杂色 灰度小于0.4位置变为0
        print(data)
        result.append(data)
        data28 = np.reshape(data, [28, 28])
        plt.imshow(data28, cmap=cm.binary)
        plt.show()

    return np.array(result)


tf.logging.set_verbosity(tf.logging.INFO)


# Our application logic will be added here


def cnn_model_fn(features, labels, mode):
    """Model function for CNN."""
    # Input Layer
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    # Convolutional Layer #1 卷积层1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,  # 使用的输入层
        filters=32,  # 使用32个过滤器
        kernel_size=[5, 5],  # 过滤器大小
        padding="same",  # 保持输出层大小于输入层大小相同
        activation=tf.nn.relu)  # 激活函数使用Relu

    # 经过该层后矩阵大小变为[n,28,28,32]

    # Pooling Layer #1 池化层1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)  # 使用最大池化层 池化过滤器大小2*2 步长为2

    # 经过该层后矩阵大小变为[n,14,14,32]

    # Convolutional Layer #2 and Pooling Layer #2  卷积层及池化层2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,  # 使用64个过滤器
        kernel_size=[5, 5],  # 过滤器大小5*5
        padding="same",  # 保持输出输入层大小相同
        activation=tf.nn.relu)  # 激活函数Relu
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)  # 最大池化层2 过滤器大小2*2 步长为2

    # 经过该层后矩阵大小变为[n,7,7,64]

    # Dense Layer 全连接层
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])  # 矩阵转换为二维[n,7*7*64]
    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)  # 将二维化的矩阵传入1024个节点的隐层，激活函数为RELU
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)  # 插入一个dropout层 丢弃40%节点

    # Logits Layer
    logits = tf.layers.dense(inputs=dropout, units=10)  # 输出层  输出类型有10类[0,1,2,3,4,5,6,7,8,9]

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
    if mode == tf.estimator.ModeKeys.TRAIN:  # 如果是训练模式
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)  # 创建一个学习率0.001的Gradient优化器
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
    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model")  # 新建自定义Estimator 传入自定义的模型函数   选定模型存放的目录

    predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": get_predict_data()},
        num_epochs=1,
        shuffle=False)

    predictions = list(mnist_classifier.predict(input_fn=predict_input_fn))
    predicted_classes = [p["classes"] for p in predictions]

    print(
        "识别结果为:    {}\n".format(predicted_classes))


if __name__ == "__main__":
    tf.app.run()

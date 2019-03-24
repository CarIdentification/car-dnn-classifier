from keras.applications.vgg19 import VGG19
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
import tensorflow as tf
import numpy as np
import os


def car_vgg_model():
    base_model = VGG19(
        weights='/Users/cai/personal/car/tfSource/car-dnn-classifier/cjt/vgg19_weights_tf_dim_ordering_tf_kernels.h5',
        include_top=False)  # 不将最顶端的三层全连接include进来

    x = base_model.output
    x = Dense(4096, activation='relu')(x)
    x = Dense(4096, activation='relu')(x)
    predictions = Dense(100, activation="softmax")(x)  # softmax函数输出100个分类

    model = Model(inputs=base_model.input, outputs=predictions)

    # 只训练最top的三层，冻结vgg19的其他层
    for layer in base_model.layers:
        layer.trainable = False

    # 编译模型
    model.compile(optimizer="rmsprop", loss="categorical_crossentropy")

    # 转化模型成estimator model ，方便tensorflow训练
    model_dir = os.path.join(os.path.abspath(__file__), "vggTrain")  # 存放在当前路径的vggTrain文件夹下面
    estimator_model = tf.keras.estimator.model_to_estimator(model, model_dir=model_dir)
    return estimator_model

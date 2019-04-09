import os

import keras.applications.mobilenet_v2 as mobilenet
import keras.layers as layers
import keras.models as models
import numpy as np
import keras.preprocessing.image as preImage
from PIL import Image as pimg

from keras.preprocessing import image


def car_mobile_model(weight_save_path=None):
    model = mobilenet.MobileNetV2(input_shape=[224, 224, 3],
                                  include_top=False
                                  )

    inp = model.input
    x = layers.GlobalAveragePooling2D()(model.layers[-1].output)
    out = layers.Dense(739, activation='softmax',
                       use_bias=True, name='predictions')(x)

    model = models.Model(inp, out)

    model.summary()

    if weight_save_path is not None:
        model.load_weights(weight_save_path, by_name=True)
    else:
        model.load_weights(
            "C:\\Users\\D\\.keras\\models\\mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5",
            by_name=True)

    for layer in model.layers[:-2]:
        layer.trainable = True

    # sgd = optimizers.SGD(lr=0.01, decay=1e-4, momentum=0.9, nesterov=True)

    # 编译模型
    model.compile(optimizer='sgd', loss="categorical_crossentropy", metrics=['accuracy'])

    return model


if __name__ == '__main__':

    weight_save_path = "G:\\Projects\\PyCharmProjects\\car-dnn-classifier\\hzd\\f\\moible_net\\car.h5"

    if os.path.exists(weight_save_path):
        mobile_car_classifier = car_mobile_model(weight_save_path)
    else:
        mobile_car_classifier = car_mobile_model()

    file_path = 'D:\\byds.jpg'

    data_loader = preImage.ImageDataGenerator(samplewise_center=True)

    img = image.load_img(file_path, target_size=(224, 224))

    x = image.img_to_array(img)

    x = np.expand_dims(x, axis=0)
    x = data_loader.standardize(x)

    g = pimg.fromarray(x[0], mode="RGB")
    g.show()

    rs = mobile_car_classifier.predict(x)
    ind = np.argpartition(rs[0], -4)[-4:]
    ind = ind[np.argsort(rs[0][ind])]
    print(str(ind))
    print(str(rs[0][ind]))


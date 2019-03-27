import os

import keras.applications.mobilenet_v2 as mobilenet
import keras.layers as layers
import keras.models as models
import keras.preprocessing.image as preImage
from keras.callbacks import TensorBoard


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


def write_epoch_num(epo_num):
    try:
        h_file = open(epoch_save_path, 'w')
        try:
            h_file.write(str(epo_num))
        finally:
            h_file.close()
    except IOError:
        print("IOError")


def read_epoch_num():
    try:
        h_file = open(epoch_save_path, 'r', 1)
        try:
            rs = h_file.readline()
            rs_num = int(rs)
            return rs_num
        finally:
            h_file.close()
    except IOError:
        print("IOError")
        return 0


if __name__ == '__main__':

    data_base_directory = "D:\\t"

    weight_save_path = "G:\\Projects\\PyCharmProjects\\car-dnn-classifier\\hzd\\f\\moible_net\\car.h5"
    epoch_save_path = "G:\\Projects\\PyCharmProjects\\car-dnn-classifier\\hzd\\mobile\\current_epoch.txt"

    data_loader = preImage.ImageDataGenerator(samplewise_center=True, validation_split=0.1, rotation_range=30,
                                              horizontal_flip=True)

    data_generator = data_loader.flow_from_directory(data_base_directory, batch_size=32,
                                                     target_size=(224, 224), subset='training')

    test_data_generator = data_loader.flow_from_directory(data_base_directory, batch_size=32,
                                                          target_size=(224, 224), subset='validation')

    if os.path.exists(weight_save_path):
        mobile_car_classifier = car_mobile_model(weight_save_path)
    else:
        mobile_car_classifier = car_mobile_model()

    board_callback = TensorBoard(write_grads=True, write_images=True, log_dir="d:\\tensor_logs_mobile")

    a = mobile_car_classifier.fit_generator(data_generator,
                                            callbacks=[board_callback],
                                            epochs=40, steps_per_epoch=36951 * 0.9 / 32,
                                            verbose=1, validation_data=test_data_generator,
                                            validation_steps=36951 * 0.1 / 32,initial_epoch=10)

    print('Save Weights')
    mobile_car_classifier.save(weight_save_path)

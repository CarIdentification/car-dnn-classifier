import os

import keras
import keras.applications.vgg19 as carv
import keras.initializers as initializers
import keras.optimizers as optimizers
import keras.preprocessing.image as preImage
import tensorflow as tf
from keras.callbacks import TensorBoard


def car_vgg_model(weight_save_path=None):
    model = carv.VGG19(input_shape=[224, 224, 3],
                       include_top=True,
                       weights=None, classes=196
                       )

    # inp = model.input
    # x = layers.Flatten(name='flatten')(model.layers[-1].output)
    # x = layers.Dense(4096, kernel_initializer="glorot_uniform", activation='relu', name='fc1')(x)
    # x = layers.Dense(4096, kernel_initializer="glorot_uniform", activation='relu', name='fc2')(x)
    # out = layers.Dense(196, kernel_initializer="glorot_uniform", activation='softmax', name='predictions')(x)
    #
    # model = models.Model(inp, out)

    # model.summary()

    model.layers[-1].name = 'car_classify_layer'
    model.layers[-1].kernel_initializer = initializers.glorot_normal()

    for a_layer in model.layers[:-1]:
        a_layer.trainable = False

    if weight_save_path is not None:
        model.load_weights(weight_save_path, by_name=True)
    else:
        model.load_weights("C:\\Users\\D\\.keras\\models\\vgg19_weights_tf_dim_ordering_tf_kernels.h5", by_name=True)

    # 只训练最top的三层，冻结vgg19的其他层
    # for layer in model.layers[:-1]:
    #     layer.trainable = False
    for layer in model.layers[:-3]:
        layer.trainable = False

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    tf.keras.backend.set_session(sess)

    sgd = optimizers.SGD(lr=0.01, decay=1e-4, momentum=0.9, nesterov=True)

    # 编译模型
    model.compile(optimizer=sgd, loss="categorical_crossentropy", metrics=['accuracy'])

    return model


data_base_directory = "G:\\Projects\\PyCharmProjects\\car-dnn-classifier\\hzd\\vgg\\stanford_car_dataset_solved\\"

weight_save_path = "G:\\Projects\\PyCharmProjects\\car-dnn-classifier\\hzd\\vgg\\car.h5"

data_loader = preImage.ImageDataGenerator(horizontal_flip=True, vertical_flip=True, zoom_range=0.25)

data_generator = data_loader.flow_from_directory(data_base_directory + "\\train", batch_size=8,
                                                 target_size=(224, 224, 3)[:-1])
test_data_generator = data_loader.flow_from_directory(data_base_directory + "\\test", batch_size=8,
                                                      target_size=(224, 224, 3)[:-1])

if os.path.exists(weight_save_path):
    vgg_car_classifier = car_vgg_model(weight_save_path)
else:
    vgg_car_classifier = car_vgg_model()

board_callback = TensorBoard(write_grads=True, write_images=True, update_freq='batch', log_dir="d:\\tensor_logs")

epoch_num = 0

while True:
    vgg_car_classifier.fit_generator(data_generator, validation_data=test_data_generator, callbacks=[board_callback],
                                     epochs=epoch_num + 1, steps_per_epoch=8144, validation_steps=8041, verbose=2,
                                     initial_epoch=epoch_num)
    epoch_num = epoch_num + 1
    print('Save Weights')
    vgg_car_classifier.save(weight_save_path)
    print('Next Round')

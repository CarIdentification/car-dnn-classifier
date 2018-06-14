import tensorflow as tf
import hzd.vgg.image_loader as il
import hzd.vgg.keras_to_tf.carvgg19 as carv
import keras
import os


def car_vgg_model():
    model = carv.car_vgg_19(input_shape=[224, 224, 3],
                            classes=196,
                            include_top=False,
                            weights='imagenet')
    model.summary()

    # 只训练最top的三层，冻结vgg19的其他层
    for layer in model.layers[:-3]:
        layer.trainable = False

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    tf.keras.backend.set_session(sess)
    # 编译模型
    model.compile(optimizer="sgd", loss="categorical_crossentropy", metrics=['accuracy'])

    return model


loader = il.ImageLoader(
    "G:\\Projects\\PyCharmProjects\\car-dnn-classifier\\hzd\\vgg\\stanford_car_dataset_solved\\train\\",
    batch_size=16)

weight_save_path = "G:\\Projects\\PyCharmProjects\\car-dnn-classifier\\hzd\\vgg\\car.h5"

if os.path.exists(weight_save_path):
    vgg_car_classifier = keras.models.load_model(weight_save_path)
else:
    vgg_car_classifier = car_vgg_model()
    vgg_car_classifier.fit(loader.x_train, keras.utils.to_categorical(loader.y_label, 196), batch_size=1,
                           epochs=1, verbose=1)
    vgg_car_classifier.save(weight_save_path)

# vgg_car_classifier.fit_generator(loader.generate(), steps_per_epoch=16, epochs=666)


while True:
    vgg_car_classifier.fit(loader.x_train, keras.utils.to_categorical(loader.y_label, 196), batch_size=16,
                           epochs=1, verbose=2, validation_split=0.01)
    loader.re_shuffle()
    vgg_car_classifier.save(weight_save_path)

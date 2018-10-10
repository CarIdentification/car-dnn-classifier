# base_model = VGG19(
#     weights='G:\\\\vgg19_weights_tf_dim_ordering_tf_kernels.h5',
#     include_top=True)
# base_model.summary()
# x = Dense(4096, activation='relu')(x)
# x = Dense(4096, activation='relu')(x)
# predictions = Dense(100, activation="softmax")(x)  # softmax函数输出100个分类
#
# model = Model(inputs=base_model.input, outputs=predictions)
#
# # 只训练最top的三层，冻结vgg19的其他层
# for layer in base_model.layers:
#     layer.trainable = False
#
# # 编译模型
# model.compile(optimizer="rmsprop", loss="categorical_crossentropy")

import keras
import hzd.vgg.keras_vgg_image_loader as il
import numpy as np

model = keras.models.load_model("G:\\Projects\\PyCharmProjects\\car-dnn-classifier\\hzd\\vgg\\car.h5")
loader = il.ImageLoader(
    "G:\\Projects\\PyCharmProjects\\car-dnn-classifier\\hzd\\vgg\\stanford_car_dataset_solved\\train\\",
    batch_size=16)
print(model.evaluate(loader.x_train, keras.utils.to_categorical(loader.y_label, 196), 16))

import os

import keras
from keras import losses
from keras.layers import Input, Dense
from keras.applications.xception import Xception
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator


# define the model
def define_model():
    car_classifier_model = Xception(weights=None, classes=196)

    # 判断之前是否有已经重新训练过的权重
    weight_file_path = '.' + os.path.sep + 'car_model_weight.h5'
    if os.path.exists(weight_file_path):
        car_classifier_model.load_weights(filepath=weight_file_path, by_name=True)
    else:
        # 加载没有最顶层的权重
        car_classifier_model.load_weights(
            filepath='./xception_weights_tf_dim_ordering_tf_kernels_notop.h5',
            by_name=True)

    # 冻结不训练的层
    for layer in car_classifier_model.layers[:-3]:
        layer.trainable = False

    # 定义优化器 ，更新和计算影响模型训练和模型输出的网络参数，使其逼近或达到最优值，从而最小化
    sgd_optimizers = keras.optimizers.SGD(lr=0.01,  # 学习率
                                          momentum=0.0,  # 动量参数
                                          decay=0.9,  # 学习衰减率
                                          nesterov=True  # 是否使用Nesterov动量
                                          )
    # 编译模型以供训练
    car_classifier_model.compile(optimizer=sgd_optimizers,  # 优化
                                 loss=losses.mean_squared_error  # 损失函数 也可称为目标函数
                                 )

    # 加上自己需要重新训练的层
    return car_classifier_model


# 准备数据
data_gen_args = dict(featurewise_center=True,  # 使输入数据集去中心化
                     featurewise_std_normalization=True,  # 将输入除以数据集的标准差以完成标准化
                     rotation_range=90.,  # 数据提升时图片随机转动的角度
                     width_shift_range=0.1,  # 数据提升时图片水平偏移的幅度
                     height_shift_range=0.1,  # 数据提升时图片竖直偏移的幅度
                     zoom_range=0.2)  # 随机缩放的幅度
train_data_gen = ImageDataGenerator(**data_gen_args)
# 数据大小比较大，不适合一次全部装入内存中，使用flow_from_directory方法按批次从硬盘读入数据并实时进行数据增强
train_generator = train_data_gen.flow_from_directory(directory='./train',  # 训练数据文件夹
                                                     target_size=(299, 299),  # 模型图片大小为 299 * 299
                                                     batch_size=8
                                                     )
validate_generator = train_data_gen.flow_from_directory(directory='./train',
                                                        target_size=(299, 299),
                                                        batch_size=8)

model = define_model()
# model.summary()

# 添加tensorboard监测
board_callback = keras.callbacks.TensorBoard(log_dir='./logs')
epoch_num = 0
while True:
    # 开始训练，在代码中修改模型
    # 逐个生成数据的batch并进行训练，生成器与模型将并行执行以提高效率
    model.fit_generator(generator=train_generator,
                        validation_data=validate_generator,
                        callbacks=[board_callback],
                        epochs=epoch_num + 1,  # 数据迭代的轮数
                        steps_per_epoch=8144  # 当生成器返回steps_per_epoch次数据时计一个epoch结束
                        )
    epoch_num += 1
    print('Save car_model_weight.h5')
    model.save('.' + os.path.sep + 'car_model_weight.h5')
    print('Next epoch')

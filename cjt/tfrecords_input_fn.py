import tensorflow as tf
#获取训练集数据

def tfRecord_input_fn(fileName):
    def parser(record):  # 预处理操作函数定义
        keys_to_features = {
            "img_raw": tf.FixedLenFeature((), tf.string, default_value=""),  # shape为 () ,tf.string 为输入的数据类型
            "label": tf.FixedLenFeature((), tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
        }
        parsed = tf.parse_single_example(record, keys_to_features)

        image = tf.image.decode_jpeg(parsed["img_raw"])
        image = tf.reshape(image, [224, 224, 3])  # 返回一个图片的张量
        image = tf.cast(image, tf.float32)  # 将图片矩阵转化成浮点数
        label = tf.cast(parsed["label"], tf.int32)
        return {"img_raw": image}, label
    def get_train_data():
        dataset = tf.data.TFRecordDataset(fileName) #读取指定的tfRecords文件
        dataset = dataset.map(parser) #对数据进行预处理
        dataset = dataset.shuffle(buffer_size=100) #buffer_size张量对应的个数进行打乱操作，提高训练泛化性
        dataset = dataset.batch(1) #单次读取几张图片
        dataset = dataset.repeat(None) #数据集重复使用次数，为空无限循环
        iterator = dataset.make_one_shot_iterator() #创建一个迭代器，循环出数据

        features , labels = iterator.get_next()
        return features , labels
    return get_train_data


features,label = tfRecord_input_fn("train_test.tfrecords")()
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    # print(sess.run(features["img_raw"]))
    print(sess.run(label))


import tensorflow as tf


def tfrecords_input_fn(file_paths, batch_size=100, num_epochs=None, shuffle=False):
    def data_generate():
        dataset = tf.data.TFRecordDataset(file_paths)
        dataset = dataset.map(parser)
        # dataset = dataset.shuffle(buffer_size=10000)  # 在训练的时候一般需要将输入数据进行顺序打乱提高训练的泛化性
        dataset = dataset.batch(batch_size)  # 单次读取的batch大小
        dataset = dataset.repeat(num_epochs)  # 数据集的重复使用次数，为空的话则无线循环
        iterator = dataset.make_one_shot_iterator()

        features, labels = iterator.get_next()
        return {"img_data": features}, labels

    def parser(record):
        keys_map = {
            "label": tf.FixedLenFeature((), tf.int64, tf.zeros([], dtype=tf.int64)),
            "img_byte": tf.FixedLenFeature((), tf.string, default_value="")
        }
        parsed_data = tf.parse_single_example(record, keys_map)
        img = tf.decode_raw(parsed_data["img_byte"], out_type=tf.uint8)

        label = parsed_data["label"]

        return img, label

    return data_generate


fea, lable = tfrecords_input_fn(["car.tfrecords"], batch_size=3)()
with tf.Session() as sess:  # 开始一个会话
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    print(sess.run(tf.reshape(fea["img_data"], [-1, 224, 224, 3])))
    print(sess.run(lable))

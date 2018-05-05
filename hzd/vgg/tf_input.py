import tensorflow as tf


def tfrecords_input_fn(file_paths, batch_size=100, num_epochs=1, shuffle=False):
    dataset = tf.data.TFRecordDataset(file_paths)


    def parser(record):
        keys_map = {
            "label": tf.FixedLenFeature((), tf.int64, tf.zeros([], dtype=tf.int64)),
            "image_byte": tf.FixedLenFeature((), tf.string, default_value="")
        }
        parsed_data = tf.parse_single_example(record, keys_map)
        img = tf.image.decode_jpeg(parsed_data["image_byte"])
        label = parsed_data["label"]

        return {"img_byte": img}, label

    dataset = dataset.map(parser)
    dataset = dataset.shuffle(buffer_size=10000)  # 在训练的时候一般需要将输入数据进行顺序打乱提高训练的泛化性
    dataset = dataset.batch(32)  # 单次读取的batch大小
    dataset = dataset.repeat(num_epochs)  # 数据集的重复使用次数，为空的话则无线循环
    iterator = dataset.make_one_shot_iterator()

    features, labels = iterator.get_next()
    return features, labels



tfrecords_input_fn(["car.tfrecords"])



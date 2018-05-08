import tensorflow as tf
import pandas as pd

TRAIN_URL = "http://download.tensorflow.org/data/iris_training.csv"
TEST_URL = "http://download.tensorflow.org/data/iris_test.csv"

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth',
                    'PetalLength', 'PetalWidth', 'Species']

...

def load_data(label_name='Species'):
    """Parses the csv file in TRAIN_URL and TEST_URL."""

    # Create a local copy of the training set.
    train_path = tf.keras.utils.get_file(fname=TRAIN_URL.split('/')[-1],
                                         origin=TRAIN_URL)
    # train_path now holds the pathname: （训练集和测试集路径） ~/.keras/datasets/iris_training.csv

    # Parse the local CSV file.（解析）
    train = pd.read_csv(filepath_or_buffer=train_path,
                        names=CSV_COLUMN_NAMES,  # list of column names
                        header=0  # ignore the first row of the CSV file.
                       )
    # train now holds a pandas DataFrame, which is data structure
    # analogous to a table.

    # 1. Assign the DataFrame's labels (the right-most column) to train_label.
    # 2. Delete (pop) the labels from the DataFrame.
    # 3. Assign the remainder of the DataFrame to train_features
    print("-")
    train_features, train_label = train, train.pop(label_name)

    # Apply the preceding logic to the test set.
    test_path = tf.keras.utils.get_file(TEST_URL.split('/')[-1], TEST_URL)
    test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
    test_features, test_label = test, test.pop(label_name)

    # Return four DataFrames.
    return (train_features, train_label), (test_features, test_label)

 # Call load_data() to parse the CSV file.
(train_feature, train_label), (test_feature, test_label) = load_data()

my_feature_columns = []
for key in train_feature.keys():
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))
#classifier 分类   estimator 估计
#使用 hidden_units 参数来定义神经网络内每个隐藏层中的神经元数量。为此参数分配一个列表。例如：
#hidden_units=[10, 10],为 hidden_units 分配的列表长度表示隐藏层的数量（在此例中为 2 个）
classifier = tf.estimator.DNNClassifier(
    feature_columns=my_feature_columns,
    hidden_units=[10, 10],
    n_classes=3,
    model_dir="./iris")
#n_classes 参数指定了神经网络可以预测的潜在值的数量。
#由于鸢尾花问题将鸢尾花品种分为 3 类，因此我们将 n_classes 设置为 3。

def train_input_fn(features, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    dataset = dataset.shuffle(buffer_size=1000).repeat(count=None).batch(batch_size)
    return dataset.make_one_shot_iterator().get_next()

classifier.train(
        input_fn=lambda:train_input_fn(train_feature, train_label, 100),
        steps=1000)
# Evaluate the model.



def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    if labels is None:
        # No labels, use only features.
        inputs = dict(features)
    else:
        inputs = (dict(features), labels)
    # Convert inputs to a tf.dataset object.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)
    # Return the read end of the pipeline.
    return dataset.make_one_shot_iterator().get_next()
eval_result = classifier.evaluate(
    input_fn=lambda:eval_input_fn(test_feature, test_label, 100)
)

print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
print("")
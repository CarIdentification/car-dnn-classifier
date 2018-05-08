# https://blog.csdn.net/xinyu3307/article/details/74643019
from joeytsai.vgg19.tfrecord_dataset_opration import read_and_decode, output_path, trans2tfRecord, file_path, \
    tfrecord2pic
import tensorflow as tf
import numpy as np
# img, label = read_and_decode(output_path+"car.tfrecords")
#
# #使用shuffle_batch可以随机打乱输入
# img_batch, label_batch = tf.train.shuffle_batch([img, label],
#                                                 batch_size=30, capacity=2000,
#                                                 min_after_dequeue=1000)
# init = tf.initialize_all_variables()
#
# with tf.Session() as sess:
#     sess.run(init)
#     threads = tf.train.start_queue_runners(sess=sess)
#     for i in range(10):
#         val, l= sess.run([img_batch, label_batch])
#         #我们也可以根据需要对val， l进行处理
#         #l = to_categorical(l, 12)
#         print(val.shape, l)
from joeytsai.vgg19.vgg19 import loss, vgg19_model, training, get_accuracy

#http://www.cnblogs.com/wktwj/p/7227544.html
if __name__ == '__main__':
   trans2tfRecord(file_path, "car", output_path)
   (image , label) = read_and_decode(output_path+"car.tfrecords")
   tfrecord2pic(output_path,"car.tfrecords",file_path+"tfrecord2pic/")
   #http://www.cnblogs.com/wktwj/p/7227544.html
   image_batches, label_batches = tf.train.batch([image, label], batch_size=16, capacity=20)

   p = vgg19_model(image_batches,0.5,4,"修改的参数")
   cost = loss(p, label_batches)
   train_op = training(cost, 0.001)

   acc = get_accuracy(p, label_batches)

   sess = tf.Session()
   init = tf.global_variables_initializer()
   sess.run(init)

   coord = tf.train.Coordinator()
   threads = tf.train.start_queue_runners(sess=sess, coord=coord)

   try:
       for step in np.arange(1000):
           print(step)
           if coord.should_stop():
               break
           _, train_acc, train_loss  = sess.run([train_op, acc, cost ])
           print("loss:{} accuracy:{}".format(train_loss, train_acc))
           print(p)
   except tf.errors.OutOfRangeError:
       print("Done!!!")
   finally:
       coord.request_stop()
   coord.join(threads)
   sess.close()
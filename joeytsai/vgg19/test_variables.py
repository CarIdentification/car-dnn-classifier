import tensorflow as tf
def test_variables(variable_name,variable):
    sess = tf.Session()
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    l = sess.run(variable)
    coord.request_stop()
    coord.join(threads)
    sess.close()
    print(" ########################  begin  {}  #########################".format(variable_name) ,"\n",l, "\n","#########################   end  {}   #########################".format(variable_name))

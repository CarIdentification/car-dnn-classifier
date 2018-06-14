#https://www.jiqizhixin.com/articles/2017-12-10-5
import tensorflow as tf
from multiprocessing import Process
from time import sleep
cluster = tf.train.ClusterSpec({
 "worker": [
     "localhost:3333",
     "localhost:3334",
 ],
 "ps": [
     "localhost:3335"
 ]
})
def parameter_server():
 with tf.device("/job:ps/task:0"):
     var = tf.Variable(0.0, name="var")
 server = tf.train.Server(cluster,
                          job_name="ps",
                          task_index=0)
 sess = tf.Session(target=server.target)
 print("Parameter server: waiting for cluster connection...")
 sess.run(tf.report_uninitialized_variables())
 print("Parameter server: cluster ready!")
 print("Parameter server: initializing variables...")
 sess.run(tf.global_variables_initializer())
 print("Parameter server: variables initialized")
 for i in range(5):
     val = sess.run(var)
     print("Parameter server: var has value %.1f" % val)
     sleep(1.0)
 print("Parameter server: blocking...")
 server.join()
def worker(worker_n):
 with tf.device("/job:ps/task:0"):
     var = tf.Variable(0.0, name="var")
 server = tf.train.Server(cluster,
                          job_name="worker",
                          task_index=worker_n)
 sess = tf.Session(target=server.target)
 print("Worker %d: waiting for cluster connection..." % worker_n)
 sess.run(tf.report_uninitialized_variables())
 print("Worker %d: cluster ready!" % worker_n)
 while sess.run(tf.report_uninitialized_variables()):
     print("Worker %d: waiting for variable initialization..." % worker_n)
     sleep(1.0)
 print("Worker %d: variables initialized" % worker_n)
 for i in range(5):
     print("Worker %d: incrementing var" % worker_n)
     sess.run(var.assign_add(1.0))
     sleep(1.0)
 print("Worker %d: blocking..." % worker_n)
 server.join()
w1_proc = Process(target=worker, args=(0, ), daemon=False)
w1_proc.start()
# for proc in [w1_proc]:
#  proc.terminate()
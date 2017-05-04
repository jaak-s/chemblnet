import unittest
import numpy as np
import tensorflow as tf
from chemblnet import SGLD, pSGLD

class SGLDTest(unittest.TestCase):
    def test_sgld_dense(self):
        tf.reset_default_graph()

        x = tf.Variable(tf.zeros(20), dtype=tf.float32)
        loss = tf.reduce_sum(tf.square(x - 10))

        sgld = SGLD(learning_rate=0.4)
        train_op_sgld = sgld.minimize(loss)

        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())

        sess.run(train_op_sgld)
        xh = sess.run(x)
        self.assertTrue(5.0 <= xh.mean() and xh.mean() <= 11.0)

    def test_sgld_sparse(self):
        tf.reset_default_graph()

        z     = tf.Variable(tf.zeros((5, 2)), dtype=tf.float32)
        idx   = tf.placeholder(tf.int32)
        zi    = tf.gather(z, idx)
        zloss = tf.square(zi - [10.0, 5.0])

        sgld = SGLD(learning_rate=0.4)
        train_op_sgld = sgld.minimize(zloss)

        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())

        self.assertTrue(np.alltrue(sess.run(z) == 0.0))

        sess.run(train_op_sgld, feed_dict={idx: 3})
        zh = sess.run(z)
        self.assertTrue(np.alltrue(zh[[0, 1, 2, 4], :] == 0.0))
        self.assertTrue(zh[3, 0] > 0)

    def test_psgld_dense(self):
        tf.reset_default_graph()

        x = tf.Variable(tf.zeros(20), dtype=tf.float32)
        loss = tf.reduce_sum(tf.square(x - 10))

        psgld = pSGLD(learning_rate=1.0)
        train_op_psgld = psgld.minimize(loss)

        sess = tf.InteractiveSession()
        sess.run(tf.global_variables_initializer())

        sess.run(train_op_psgld)
        xh = sess.run(x)

if __name__ == '__main__':
    unittest.main()

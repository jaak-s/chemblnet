import tensorflow as tf
import chemblnet as cn
import numpy as np

beta = cn.NormalGammaUni("beta", [100, 5])

sess = tf.Session()
sess.run(tf.initialize_all_variables())
print("beta.prec_div()   = %f" % sess.run(beta.prec_div()))
print("beta.normal_div() = %f" % sess.run(beta.normal_div()))
print("beta.normal_div() = %f (on mean 0, logvar 0)" % sess.run(beta.normal_div(),
        feed_dict = {beta.mean: np.zeros((100, 5)), beta.logvar: np.zeros((100, 5))}
    ))


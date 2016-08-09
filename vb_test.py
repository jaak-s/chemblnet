import tensorflow as tf
import vbutils as vb
import numpy as np

beta = vb.NormalGammaUni("beta", [100, 5])

sess = tf.Session()
sess.run(tf.initialize_all_variables())
sess.run(beta.prec_negdiv())
sess.run(beta.normal_negdiv())

sess.run(beta.normal_negdiv(), feed_dict = {beta.mean: np.zeros((100, 5)), beta.var: np.zeros((100, 5))})


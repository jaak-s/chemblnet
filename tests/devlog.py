import vbutils as vb
import tensorflow as tf
import numpy as np

sess = tf.Session()
x = vb.NormalGammaUni("x", [3, 2])

## new:
sess.run([x.normal_div(), x.prec_div()], feed_dict={x.mean: np.array([[1.,2], [3,4],[5,6]]), x.logvar: np.log(np.array([[0.5, 1.5], [2.5,3.5], [4.5, 5.5]])), x.prec_b: [1.0, 2.0], x.prec_a: [0.5, 0.6] })

## old:
sess.run([x.normal_div(), x.prec_div()], feed_dict={x.mean: np.array([[1.,2], [3,4],[5,6]]), x.var: np.array([[0.5, 1.5], [2.5,3.5], [4.5, 5.5]]), x.prec_b: [1.0, 2.0], x.prec_a: [0.5, 0.6] })
# [24.350817, 1.4889445]


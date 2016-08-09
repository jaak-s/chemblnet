import tensorflow as tf
import numpy as np
import scipy.special

## outputs negative D_KL( G(alpha, beta) || G(n,m) )
## where alpha and beta are tf variables (tensors) and n and m are constants
def gammaPrior(alpha, beta, n, m):
  return - (alpha - n)*tf.digamma(alpha) + tf.lgamma(alpha) - scipy.special.gammaln(n) - n * (tf.log(beta) - np.log(m)) - alpha * (m / beta - 1.0)

class NormalGammaUni:
    def __init__(self, name, shape, initial_var = 4.0, initial_prec_a = 0.1, initial_prec_b = 0.1, a0 = 0.1, b0 = 0.1):
        self.mean   = tf.get_variable(name=name + ".mean", shape=shape, initializer=tf.contrib.layers.xavier_initializer(), dtype = tf.float32)
        self.var    = tf.Variable(initial_var * np.ones(shape),      name = name + ".var", dtype = tf.float32)
        self.prec_a = tf.Variable(initial_prec_a * np.ones(shape[-1]), name = name + ".prec_a", dtype = tf.float32)
        self.prec_b = tf.Variable(initial_prec_b * np.ones(shape[-1]), name = name + ".prec_b", dtype = tf.float32)
        self.prec   = tf.div(self.prec_a, self.prec_b)
        self.a0     = a0
        self.b0     = b0
        self.shape  = shape

    def prec_div(self):
        return - tf.reduce_sum(gammaPrior(self.prec_a, self.prec_b, self.a0, self.b0))

    ## outputs E_q[ log N( x | 0, prec^-1) ] + Entropy(q(x))
    ## where x is the normally distributed variable
    def normal_div(self):
        regul = tf.mul(self.prec, tf.reduce_sum(tf.square(self.mean), 0) + tf.reduce_sum(self.var, 0))

        return (tf.reduce_sum(regul) / 2.0
                - self.shape[0] / 2.0 * tf.reduce_sum(tf.digamma(self.prec_a) - tf.log(self.prec_b))
                - tf.reduce_sum(tf.log(self.var)) / 2.0
               )


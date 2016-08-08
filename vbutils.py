import tensorflow as tf
import numpy as np
import scipy.special

## outputs negative D_KL( G(alpha, beta) || G(n,m) )
## where alpha and beta are tf variables (tensors) and n and m are constants
def gammaPrior(alpha, beta, n, m):
  return - (alpha - n)*tf.digamma(alpha) + tf.lgamma(alpha) - scipy.special.gammaln(n) - n * (tf.log(beta) - np.log(m)) - alpha * (m / beta - 1.0)

class NormalGammaUni:
    def __init__(self, name, shape, initial_var = 4.0, initial_prec_a = 0.1, initial_prec_b = 0.1):
        self.mean   = tf.get_variable(name=name + ".mean", shape=shape, initializer=tf.contrib.layers.xavier_initializer())
        self.var    = tf.Variable(initial_var * np.ones(shape),      name = name + ".var")
        self.prec_a = tf.Variable(initial_prec_a * np.ones(shape[-1]), name = name + ".prec_a")
        self.prec_b = tf.Variable(initial_prec_b * np.ones(shape[-1]), name = name + ".prec_b")

    def prec_negdiv(self, n, m):
        return gammaPrior(self.prec_a, self.prec_b, n, m)

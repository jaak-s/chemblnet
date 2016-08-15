import tensorflow as tf
import numpy as np
import scipy.special

## outputs negative D_KL( G(alpha, beta) || G(n,m) )
## where alpha and beta are tf variables (tensors) and n and m are constants
def gammaPrior(alpha, beta, n, m):
  return - (alpha - n)*tf.digamma(alpha) + tf.lgamma(alpha) - scipy.special.gammaln(n) - n * (tf.log(beta) - np.log(m)) - alpha * (m / beta - 1.0)

class NormalGammaUni:
    def __init__(self, name, shape, initial_stdev = 2.0, initial_prec_a = 0.1, initial_prec_b = 0.1, a0 = 0.1, b0 = 0.1):
        with tf.variable_scope(name) as scope:
            self.mean   = tf.get_variable(name="mean", shape=shape, initializer=tf.contrib.layers.xavier_initializer(), dtype = tf.float32)
            #self.var    = tf.Variable(initial_var * np.ones(shape),      name = name + ".var", dtype = tf.float32)
            self.logvar = tf.Variable(np.log(initial_stdev**2.0) * np.ones(shape), name = "logvar", dtype = tf.float32)
            self.prec_a = tf.Variable(initial_prec_a * np.ones(shape[-1]), name = "prec_a", dtype = tf.float32)
            self.prec_b = tf.Variable(initial_prec_b * np.ones(shape[-1]), name = "prec_b", dtype = tf.float32)
            self.prec   = tf.div(self.prec_a, self.prec_b, name = "prec")
            self.var    = tf.exp(self.logvar, name = "var")
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
                - tf.reduce_sum(self.logvar) / 2.0
               )

    def normal_div_partial(self, pmean, plogvar, n):
        prop  = self.shape[0] / float(n)
        regul = prop * tf.mul(self.prec, tf.reduce_sum(tf.square(pmean), 0) + tf.reduce_sum(tf.exp(plogvar), 0))
        return (tf.reduce_sum(regul) / 2.0
                - self.shape[0] / 2.0 * tf.reduce_sum(tf.digamma(self.prec_a) - tf.log(self.prec_b))
                - tf.reduce_sum(plogvar) / 2.0
               )

def embedding_lookup_sparse_sumexp(params, sp_ids,
                                   name=None):
    segment_ids = sp_ids.indices[:, 0]
    if segment_ids.dtype != tf.int32:
      segment_ids = tf.cast(segment_ids, tf.int32)

    ids = sp_ids.values
    ids, idx = tf.unique(ids)

    embeddings = tf.nn.embedding_lookup(params, ids)
    embeddings = tf.exp(embeddings)
    embeddings = tf.sparse_segment_sum(embeddings, idx, segment_ids,
                                       name=name)

    return embeddings

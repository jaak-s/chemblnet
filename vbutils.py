import tensorflow as tf
import numpy as np
import scipy.special

## outputs negative D_KL( G(alpha, beta) || G(n,m) )
## where alpha and beta are tf variables (tensors) and n and m are constants
def gammaPrior(alpha, beta, n, m):
  return - (alpha - n)*tf.digamma(alpha) + tf.lgamma(alpha) - scipy.special.gammaln(n) - n * (tf.log(beta) - np.log(m)) - alpha * (m / beta - 1.0)


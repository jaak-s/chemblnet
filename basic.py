import tensorflow as tf
import scipy.io
import numpy as np
import lookup_ops

def csr2indices(csr):
  counts  = (csr.indptr[1:] - csr.indptr[0:-1]).reshape(-1, 1)
  indices = np.zeros((csr.nnz, 1), dtype = np.int64)
  for i in range(csr.shape[0]):
    indices[ csr.indptr[i] : csr.indptr[i+1], 0 ] = i
  shape   = [0, 0]
  return indices, shape, csr.indices.astype(np.int64, copy=False)

class Data:
  def __init__(self, Xtr, Xte, Ytr, Yte):
    self.Xtrain = Xtr
    self.Ytrain = Ytr.reshape(-1, 1)
    self.Xtest  = Xte
    self.Ytest  = Yte.reshape(-1, 1)
    self.Nfeat  = Xtr.shape[1]

def split_train_test(X, Y, ratio):
    nrow  = X.shape[0]
    ntest = int(round(nrow * ratio))
    rperm = np.random.permutation(nrow)
    train = rperm[ntest:]
    test  = rperm[0:ntest]
    return Data(X[train], X[test], Y[train], Y[test])

label = scipy.io.mmread("chembl-IC50-346targets.mm")
data  = scipy.io.mmread("chembl-IC50-compound-feat.mm")
Y167 = label.tocsc()[:,167].tocoo()
Ya = Y167.tocsc()[Y167.nonzero()[0]].data
Xa = data.tocsc()[Y167.nonzero()[0]].tocsr()
chembl = split_train_test(Xa, Ya, 0.2)
print("Data loaded.")

W = tf.Variable(tf.truncated_normal([chembl.Nfeat, 1], stddev=1/500.0))
sp_indices = tf.placeholder(tf.int64)
sp_shape   = tf.placeholder(tf.int64)
sp_ids_val = tf.placeholder(tf.int64)
sp_ids     = tf.SparseTensor(sp_indices, sp_ids_val, sp_shape)
y          = tf.nn.embedding_lookup_sparse(W, sp_ids, None, combiner = "sum")
ysq        = lookup_ops.embedding_lookup_sparse_sq(W, sp_ids, None, combiner = "sum")

sess = tf.Session()
sess.run(tf.initialize_all_variables())

test_indices, test_shape, test_ids_val = csr2indices(Xa[0:2])

y_values = sess.run(y, feed_dict={
  sp_indices: test_indices, #np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1]]),  # 3 entries in minibatch entry 0, 2 entries in entry 1.
  sp_shape: test_shape, #[2, 3],  # batch size: 2, max index: 2 (so index count == 3)
  sp_ids_val: test_ids_val}) #np.array([53, 87, 101, 34, 98])})

print(y_values)

ysq_values = sess.run(ysq, feed_dict={
  sp_indices: test_indices, #np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1]]),  # 3 entries in minibatch entry 0, 2 entries in entry 1.
  sp_shape: test_shape, #[2, 3],  # batch size: 2, max index: 2 (so index count == 3)
  sp_ids_val: test_ids_val}) #np.array([53, 87, 101, 34, 98])})

print(ysq_values)

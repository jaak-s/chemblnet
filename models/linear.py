import tensorflow as tf
import scipy.io
import numpy as np
import chemblnet as cn

label = scipy.io.mmread("chembl-IC50-346targets.mm")
data  = scipy.io.mmread("chembl-IC50-compound-feat.mm")
# 109, 167, 168, 204, 214, 215

Xtr, Ytr, Xte, Yte = cn.make_target_col(data, label, 168, 0.2)
Nfeat = Xtr.shape[1]

print("Data loaded.\nNtrain = %d\nNtest = %d" % (Ytr.shape[0], Yte.shape[0]))

print("St. deviation:   %f" % np.std( Yte ))

from scipy.sparse.linalg import lsqr
for damp in [1.0, 4.0, 10.0]:
  solution = lsqr(Xtr, Ytr, damp = damp, atol=1e-7, btol=1e-7)                     
  print("RMSE (reg= %.1f): %f" % (damp, np.sqrt(np.mean((Xte.dot(solution[0]) - Yte)**2))) )

# linear model for sparse input
W = tf.Variable(tf.truncated_normal([Nfeat, 1], stddev=1/500.0))
b = tf.Variable(tf.zeros([1]))
W_reg      = tf.placeholder("float")
y          = tf.placeholder("float", shape=[None, 1])
sp_indices = tf.placeholder(tf.int64)
sp_shape   = tf.placeholder(tf.int64)
sp_ids_val = tf.placeholder(tf.int64)
sp_ids     = tf.SparseTensor(sp_indices, sp_ids_val, sp_shape)
y_pred     = tf.nn.embedding_lookup_sparse(W, sp_ids, None, combiner = "sum") + b
y_loss     = tf.reduce_mean(tf.square(y - y_pred))
l2_reg     = W_reg * tf.nn.l2_loss(W)
loss       = l2_reg + y_loss

# Use the adam optimizer
train_op   = tf.train.AdamOptimizer(3e-3).minimize(loss)

# test set
Xte_indices, Xte_shape, Xte_ids_val = cn.csr2indices(Xte)
Yte = Yte.reshape(-1, 1)
Xtr_indices, Xtr_shape, Xtr_ids_val = cn.csr2indices(Xtr)
Ytr2 = Ytr.reshape(-1, 1)

batch_size = 32

with tf.Session() as sess:
  sess.run(tf.initialize_all_variables())

  for epoch in range(300):
    rIdx = np.random.permutation(Ytr.shape[0])

    ## mini-batch loop
    for start in np.arange(0, Ytr.shape[0], batch_size):
      if start + batch_size > Ytr.shape[0]:
        break
      idx = rIdx[start : start + batch_size]
      indices, shape, ids_val = cn.csr2indices(Xtr[idx,:])
      y_batch = Ytr[idx].reshape(-1, 1)
      sess.run(train_op, feed_dict={sp_indices: indices, sp_shape: shape, sp_ids_val: ids_val, y: y_batch, W_reg: 0.1})

    ## epoch's Ytest error
    if epoch % 10 == 0:
      test_error  = sess.run(y_loss, feed_dict = {sp_indices: Xte_indices, sp_shape: Xte_shape, sp_ids_val: Xte_ids_val, y: Yte})
      train_error = sess.run(y_loss, feed_dict = {sp_indices: Xtr_indices, sp_shape: Xtr_shape, sp_ids_val: Xtr_ids_val, y: Ytr2})
      W_l2 = sess.run(tf.nn.l2_loss(W))
      print("%3d. RMSE(test) = %.5f  RMSE(train) = %.5f  ||W|| = %.5f" % (epoch, np.sqrt(test_error), np.sqrt(train_error), np.sqrt(W_l2) ))




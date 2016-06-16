import tensorflow as tf
import scipy.io
import numpy as np
import chembl_data as cd

label = scipy.io.mmread("chembl-IC50-346targets.mm")
data  = scipy.io.mmread("chembl-IC50-compound-feat.mm")
# 109, 167, 168, 204, 214, 215

Xtr, Ytr, Xte, Yte = cd.make_target_col(data, label, 168, 0.2)
Nfeat = Xtr.shape[1]

print("Data loaded.\nNtrain = %d\nNtest = %d" % (Ytr.shape[0], Yte.shape[0]))

print("St. deviation:   %f" % np.std( Yte ))

from scipy.sparse.linalg import lsqr
for damp in [1.0, 4.0, 10.0]:
  solution = lsqr(Xtr, Ytr, damp = damp, atol=1e-7, btol=1e-7)                     
  print("RMSE (reg= %.1f): %f" % (damp, np.sqrt(np.mean((Xte.dot(solution[0]) - Yte)**2))) )

# linear model for sparse input
W1 = tf.Variable(tf.truncated_normal([Nfeat, 20], stddev=1/500.0))
b1 = tf.Variable(tf.zeros([20]))
W2 = tf.Variable(tf.truncated_normal([20, 1], stddev=1/10.0))
b2 = tf.Variable(tf.zeros([1]))
lambda_reg = tf.placeholder("float")
y          = tf.placeholder("float", shape=[None, 1])
sp_indices = tf.placeholder(tf.int64)
sp_shape   = tf.placeholder(tf.int64)
sp_ids_val = tf.placeholder(tf.int64)
sp_ids     = tf.SparseTensor(sp_indices, sp_ids_val, sp_shape)
h1         = tf.nn.relu(tf.nn.embedding_lookup_sparse(W1, sp_ids, None, combiner = "sum") + b1)
y_pred     = tf.matmul(h1, W2) + b2
y_loss     = tf.reduce_mean(tf.square(y - y_pred))
l2_reg     = lambda_reg * tf.nn.l2_loss(W1) + lambda_reg * tf.nn.l2_loss(W2)
loss       = l2_reg + y_loss

# Use the adam optimizer
train_op   = tf.train.AdamOptimizer(3e-3).minimize(loss)

# test set
Xte_indices, Xte_shape, Xte_ids_val = cd.csr2indices(Xte)
Yte = Yte.reshape(-1, 1)

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
      indices, shape, ids_val = cd.csr2indices(Xtr[idx,:])
      y_batch = Ytr[idx].reshape(-1, 1)
      sess.run(train_op, feed_dict={sp_indices: indices, sp_shape: shape, sp_ids_val: ids_val, y: y_batch, lambda_reg: 0.5})

    ## epoch's Ytest error
    if epoch % 10 == 0:
      test_error = sess.run(y_loss, feed_dict = {sp_indices: Xte_indices, sp_shape: Xte_shape, sp_ids_val: Xte_ids_val, y: Yte})
      W1_l2 = sess.run(tf.nn.l2_loss(W1))
      W2_l2 = sess.run(tf.nn.l2_loss(W2))
      print(epoch, test_error, W1_l2, W2_l2)




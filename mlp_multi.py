import tensorflow as tf
import scipy.io
import numpy as np
import chembl_data as cd

label = scipy.io.mmread("chembl-IC50-346targets.mm")
X     = scipy.io.mmread("chembl-IC50-compound-feat.mm").tocsr()
# 109, 167, 168, 204, 214, 215

#Xtr, Ytr, Xte, Yte = cd.make_target_col(data, label, 168, 0.2)
Ytrain, Ytest = cd.make_train_test(label, 0.2)
Ytrain = Ytrain.tocsr()
Ytest  = Ytest.tocsr()
Nfeat  = X.shape[1]
Nprot  = Ytrain.shape[1]

print("St. deviation:   %f" % np.std( Ytest.data ))

h1_size    = 100
batch_size = 512
reg        = 0.02
lrate0     = 0.08
lrate_decay = 1.0 #0.986

## variables for the model
W1 = tf.Variable(tf.truncated_normal([Nfeat, h1_size], stddev=1/500.0))
b1 = tf.Variable(tf.zeros([h1_size]))
W2 = tf.Variable(tf.truncated_normal([Nprot, h1_size], stddev=1/50.0))
b2 = tf.Variable(tf.zeros([Nprot]))

## inputs
y_val      = tf.placeholder(tf.float32)
y_idx_prot = tf.placeholder(tf.int64)
y_idx_comp = tf.placeholder(tf.int64)
sp_indices = tf.placeholder(tf.int64)
sp_shape   = tf.placeholder(tf.int64)
sp_ids_val = tf.placeholder(tf.int64)

## regularization parameter
lambda_reg = tf.placeholder(tf.float32)
learning_rate = tf.placeholder(tf.float32)

## model setup
sp_ids     = tf.SparseTensor(sp_indices, sp_ids_val, sp_shape)
h1         = tf.nn.elu(tf.nn.embedding_lookup_sparse(W1, sp_ids, None, combiner = "sum") + b1)

h1e        = tf.nn.embedding_lookup(h1, y_idx_comp)
W2e        = tf.nn.embedding_lookup(W2, y_idx_prot)
y_pred     = tf.squeeze(tf.batch_matmul(h1e, W2e, adj_y=True), [1, 2]) + tf.nn.embedding_lookup(b2, tf.squeeze(y_idx_prot, [1]))

y_loss     = tf.reduce_sum(tf.square(y_val - y_pred))
l2_reg     = lambda_reg * tf.nn.l2_loss(W1) + lambda_reg * tf.nn.l2_loss(W2)
loss       = l2_reg + y_loss / np.float32(batch_size)

# Use the adam optimizer
#train_op   = tf.train.AdamOptimizer(0.005).minimize(loss)
#train_op   = tf.train.RMSPropOptimizer(0.005, momentum = 0.9).minimize(loss)
train_op   = tf.train.AdagradOptimizer(learning_rate).minimize(loss)

def select_rows(X, row_idx):
  Xtmp = X[row_idx]
  indices = np.zeros((Xtmp.nnz, 1), dtype = np.int64)
  for i in range(row_idx.shape[0]):
    indices[ Xtmp.indptr[i] : Xtmp.indptr[i+1], 0 ] = i
  shape   = [0, 0]
  return indices, shape, Xtmp.indices.astype(np.int64, copy=False)

def select_y(X, row_idx):
  Xtmp = X[row_idx]
  indices = np.zeros((Xtmp.nnz, 1), dtype = np.int64)
  for i in np.arange(row_idx.shape[0]):
    indices[ Xtmp.indptr[i] : Xtmp.indptr[i+1], 0 ] = i
  return indices, [0, 0], Xtmp.indices.astype(np.int64, copy=False).reshape(-1, 1), Xtmp.data.astype(np.float32, copy=False)

Xi, Xs, Xv = select_rows(X, np.arange(X.shape[0]))
Yte_idx_comp, Yte_shape, Yte_idx_prot, Yte_val = select_y(Ytest, np.arange(Ytest.shape[0]))

# sess.run(h1e, feed_dict={sp_indices: bx_indices, sp_shape: [0, 0], sp_ids_val: bx_ids_val, y_idx_comp : by_idx_comp })
# sess.run(W2e, feed_dict={sp_indices: bx_indices, sp_shape: [0, 0], sp_ids_val: bx_ids_val, y_idx_comp : by_idx_comp, y_idx_prot: by_idx_prot })
# sess.run(y_pred, feed_dict={y_val: by_val, y_idx_prot: by_idx_prot, y_idx_comp: by_idx_comp, sp_indices: bx_indices, sp_shape: [0, 0], sp_ids_val: bx_ids_val })
# sess.run(y_loss, feed_dict={y_val: by_val, y_idx_prot: by_idx_prot, y_idx_comp: by_idx_comp, sp_indices: bx_indices, sp_shape: [0, 0], sp_ids_val: bx_ids_val, y_val: by_val })

with tf.Session() as sess:
  sess.run(tf.initialize_all_variables())

  for epoch in range(300):
    lrate = lrate0 * lrate_decay**epoch
    rIdx = np.random.permutation(Ytrain.shape[0])

    ## mini-batch loop
    for start in np.arange(0, Ytrain.shape[0], batch_size):
      if start + batch_size > Ytrain.shape[0]:
        break
      idx = rIdx[start : start + batch_size]
      bx_indices, bx_shape, bx_ids_val           = select_rows(X, idx)
      by_idx_comp, by_shape, by_idx_prot, by_val = select_y(Ytrain, idx)

      sess.run(train_op, feed_dict={sp_indices: bx_indices,
                                    sp_shape:   bx_shape,
                                    sp_ids_val: bx_ids_val,
                                    y_idx_comp: by_idx_comp,
                                    y_idx_prot: by_idx_prot,
                                    y_val:      by_val,
                                    lambda_reg: reg,
                                    learning_rate: lrate})

    ## epoch's Ytest error
    if epoch % 1 == 0:
      test_sse = sess.run(y_loss,  feed_dict = {sp_indices: Xi,
                                                 sp_shape:   Xs,
                                                 sp_ids_val: Xv,
                                                 y_idx_comp: Yte_idx_comp,
                                                 y_idx_prot: Yte_idx_prot,
                                                 y_val:      Yte_val})
      #train_rmse = sess.run(tf.sqrt(y_loss), feed_dict = {sp_indices: Xtr_indices, sp_shape: Xtr_shape, sp_ids_val: Xtr_ids_val, y: Ytr2})
      W1_l2 = sess.run(tf.nn.l2_loss(W1))
      W2_l2 = sess.run(tf.nn.l2_loss(W2))
      test_rmse = np.sqrt( test_sse / Yte_val.shape[0])
      #print("%3d. RMSE(test) = %.5f  RMSE(train) = %.5f  ||W1|| = %.5f ||W2|| = %.5f" % (epoch, test_rmse, train_rmse, np.sqrt(W1_l2), np.sqrt(W2_l2)) )
      print("%3d. RMSE(test) = %.5f  ||W1|| = %.5f ||W2|| = %.5f" % (epoch, test_rmse, np.sqrt(W1_l2), np.sqrt(W2_l2)) )


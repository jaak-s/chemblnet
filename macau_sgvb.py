import tensorflow as tf
import scipy.io
import numpy as np
import chembl_data as cd
import vbutils as vb

label = scipy.io.mmread("chembl-IC50-346targets.mm")
X     = scipy.io.mmread("chembl-IC50-compound-feat.mm").tocsr()
# 109, 167, 168, 204, 214, 215

Ytrain, Ytest = cd.make_train_test(label, 0.2)
Ytrain = Ytrain.tocsr()
Ytest  = Ytest.tocsr()
Nfeat  = X.shape[1]
Ncomp  = Ytrain.shape[0]
Nprot  = Ytrain.shape[1]
print("St. deviation:   %f" % np.std( Ytest.data ))

# learning parameters
Y_prec      = 5.0
h1_size     = 32

batch_size  = 512
lrate0      = 1e-5
lrate_decay = 1.0 #0.986

## inputs
y_val      = tf.placeholder(tf.float32)
y_idx_prot = tf.placeholder(tf.int64)
y_idx_comp = tf.placeholder(tf.int64)
x_indices  = tf.placeholder(tf.int64)
x_shape    = tf.placeholder(tf.int64)
x_ids_val  = tf.placeholder(tf.int64)
x_idx_comp = tf.placeholder(tf.int64) ## true compound indices

learning_rate = tf.placeholder(tf.float32)

## ratio of total training points to mini-batch training points, for the current batch
tb_ratio = tf.placeholder(tf.float32)

## model
beta  = vb.NormalGammaUni("beta", shape = [Nfeat, h1_size], initial_var = 0.01)
Z     = vb.NormalGammaUni("Z",    shape = [Ncomp, h1_size], initial_var = 1.0)
V     = vb.NormalGammaUni("V",    shape = [Nprot, h1_size], initial_var = 1.0)
global_mean = tf.Variable(Ytrain.data.mean(), dtype=tf.float32)

## expected data log likelihood
sp_ids  = tf.SparseTensor(x_indices, x_ids_val, x_shape)

## means
Zmean_b = tf.nn.embedding_lookup(Z.mean, x_idx_comp)
h1      = tf.nn.embedding_lookup_sparse(beta.mean, sp_ids, None, combiner = "sum") + Zmean_b
h1_b    = tf.nn.embedding_lookup(h1, y_idx_comp)
Vmean_b = tf.nn.embedding_lookup(V.mean, y_idx_prot)
y_pred  = global_mean + tf.squeeze(tf.batch_matmul(h1_b, Vmean_b, adj_y=True), [1, 2])
#y_pred = tf.squeeze(tf.batch_matmul(h1_b, Vmean_b, adj_y=True), [1, 2]) + tf.nn.embedding_lookup(b2, tf.squeeze(y_idx_prot, [1]))
y_loss  = Y_prec / 2.0 * tf.reduce_sum(tf.square(y_val - y_pred))

## variance
Zvar_b  = tf.nn.embedding_lookup(Z.var, x_idx_comp)
h1var   = tf.nn.embedding_lookup_sparse(beta.var, sp_ids, None, combiner = "sum") + Zvar_b
h1var_b = tf.nn.embedding_lookup(h1var, y_idx_comp)
Vvar_b  = tf.nn.embedding_lookup(V.var, y_idx_prot)

E_usq   = tf.add(h1var_b, tf.square(h1_b))
y_var1  = Y_prec / 2.0 * tf.reduce_sum(tf.squeeze(tf.batch_matmul(E_usq, Vvar_b, adj_y=True), [1, 2]))
y_var2  = Y_prec / 2.0 * tf.reduce_sum(tf.squeeze(tf.batch_matmul(h1var_b, tf.square(Vmean_b), adj_y=True), [1, 2]))

L_D     = tb_ratio * (y_loss + y_var1 + y_var2)
L_prior = beta.prec_div() + Z.prec_div() + V.prec_div() + beta.normal_div() + Z.normal_div() + V.normal_div()
loss    = L_D + L_prior

train_op = tf.train.AdagradOptimizer(learning_rate).minimize(loss)

######################################################

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

#X_ids      = np.arange(X.shape[0])
## debugging
rIdx = np.random.permutation(Ytrain.shape[0])
#idx  = rIdx[10 : 12]

#bx_indices, bx_shape, bx_ids_val           = select_rows(X, idx)
#by_idx_comp, by_shape, by_idx_prot, by_val = select_y(Ytrain, idx)

Xi, Xs, Xv = select_rows(X, np.arange(X.shape[0]))
Xindices   = np.arange(X.shape[0])
Yte_idx_comp, Yte_shape, Yte_idx_prot, Yte_val = select_y(Ytest, np.arange(Ytest.shape[0]))

# sess = tf.Session()
# sess.run(tf.initialize_all_variables())
# sess.run(y_pred, feed_dict={x_indices:  bx_indices,
#                             x_shape:    bx_shape,
#                             x_ids_val:  bx_ids_val,
#                             x_idx_comp: idx,
#                             y_idx_comp: by_idx_comp,
#                             y_idx_prot: by_idx_prot,
#                             y_val:      by_val
#                             })
# 
# sess.run(L_D, feed_dict={x_indices:  bx_indices,
#                          x_shape:    bx_shape,
#                          x_ids_val:  bx_ids_val,
#                          x_idx_comp: idx,
#                          y_idx_comp: by_idx_comp,
#                          y_idx_prot: by_idx_prot,
#                          y_val:      by_val,
#                          tb_ratio:   Ytrain.nnz / float(by_idx_comp.shape[0])
#                          })

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

      sess.run(train_op, feed_dict={x_indices:  bx_indices,
                                    x_shape:    bx_shape,
                                    x_ids_val:  bx_ids_val,
                                    y_idx_comp: by_idx_comp,
                                    y_idx_prot: by_idx_prot,
                                    y_val:      by_val,
                                    x_idx_comp: idx,
                                    learning_rate: lrate})

    ## epoch's Ytest error
    if epoch % 1 == 0:
      test_sse = sess.run(tf.reduce_sum(tf.square(y_val - y_pred)),
                          feed_dict = {x_indices:  Xi,
                                       x_shape:    Xs,
                                       x_ids_val:  Xv,
                                       y_idx_comp: Yte_idx_comp,
                                       y_idx_prot: Yte_idx_prot,
                                       y_val:      Yte_val,
                                       x_idx_comp: Xindices})
      #W1_l2 = sess.run(tf.nn.l2_loss(W1))
      #W2_l2 = sess.run(tf.nn.l2_loss(W2))
      test_rmse = np.sqrt( test_sse / Yte_val.shape[0])
      print("%3d. RMSE(test) = %.5f" % (epoch, test_rmse) )

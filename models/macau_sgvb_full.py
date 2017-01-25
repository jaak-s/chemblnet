import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--side",  type=str,   help="side information", default = "chembl-IC50-compound-feat.mm")
parser.add_argument("--y",     type=str,   help="matrix", default = "chembl-IC50-346targets.mm")
args = parser.parse_args()

import tensorflow as tf
import scipy.io
import numpy as np
import scipy as sp
import chemblnet.vbutils as vb
import numbers

def make_train_test(Y, ntest, seed = None):
    if type(Y) not in [sp.sparse.coo.coo_matrix, sp.sparse.csr.csr_matrix, sp.sparse.csc.csc_matrix]:
        raise ValueError("Unsupported Y type: %s" + type(Y))
    if not isinstance(ntest, numbers.Real) or ntest < 0:
        raise ValueError("ntest has to be a non-negative number (number or ratio of test samples).")
    Y = Y.tocoo(copy = False)
    if ntest < 1:
        ntest = Y.nnz * ntest
    if seed is not None:
        np.random.seed(seed)
    ntest = int(round(ntest))
    rperm = np.random.permutation(Y.nnz)
    train = rperm[ntest:]
    test  = rperm[0:ntest]
    Ytrain = sp.sparse.coo_matrix( (Y.data[train], (Y.row[train], Y.col[train])), shape=Y.shape )
    Ytest  = sp.sparse.coo_matrix( (Y.data[test],  (Y.row[test],  Y.col[test])),  shape=Y.shape )
    return Ytrain, Ytest

label = scipy.io.mmread(args.y)
X     = scipy.io.mmread(args.side).tocsr()
# 109, 167, 168, 204, 214, 215

Ytrain, Ytest = make_train_test(label, 0.2)
Ytrain = Ytrain.tocsr()
Ytest  = Ytest.tocsr()
Nfeat  = X.shape[1]
Ncomp  = Ytrain.shape[0]
Nprot  = Ytrain.shape[1]
print("St. deviation:   %f" % np.std( Ytest.data ))

# learning parameters
Y_prec      = 5.0
h1_size     = 30

batch_size  = 512

## inputs
y_val      = tf.placeholder(tf.float32)
y_idx_prot = tf.placeholder(tf.int64)
y_idx_comp = tf.placeholder(tf.int64)
x_indices  = tf.placeholder(tf.int64)
x_shape    = tf.placeholder(tf.int64)
x_ids_val  = tf.placeholder(tf.int64)
x_idx_comp = tf.placeholder(tf.int64) ## true compound indices

learning_rate = tf.placeholder(tf.float32, name = "learning_rate")

## ratio of total training points to mini-batch training points, for the current batch
tb_ratio = tf.placeholder(tf.float32, name = "tb_ratio")

## model
beta  = vb.NormalGammaUni("beta", shape = [Nfeat, h1_size], initial_stdev = 0.05)
Z     = vb.NormalGammaUni("Z",    shape = [Ncomp, h1_size], initial_stdev = 1.0)
V     = vb.NormalGammaUni("V",    shape = [Nprot, h1_size], initial_stdev = 1.0)
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
Zvar_b  = tf.exp(tf.nn.embedding_lookup(Z.logvar, x_idx_comp))
h1var   = vb.embedding_lookup_sparse_sumexp(beta.logvar, sp_ids) + Zvar_b
#h1var   = tf.nn.embedding_lookup_sparse(beta.var, sp_ids, None, combiner = "sum") + Zvar_b
h1var_b = tf.nn.embedding_lookup(h1var, y_idx_comp)
Vvar_b  = tf.exp(tf.nn.embedding_lookup(V.logvar, y_idx_prot))

E_usq   = tf.add(h1var_b, tf.square(h1_b))
y_var1  = Y_prec / 2.0 * tf.reduce_sum(tf.squeeze(tf.batch_matmul(E_usq, Vvar_b, adj_y=True), [1, 2]))
y_var2  = Y_prec / 2.0 * tf.reduce_sum(tf.squeeze(tf.batch_matmul(h1var_b, tf.square(Vmean_b), adj_y=True), [1, 2]))

L_D     = tb_ratio * (y_loss + y_var1 + y_var2)
L_prior = beta.prec_div() + Z.prec_div() + V.prec_div() + beta.normal_div() + Z.normal_div() + V.normal_div()
loss    = L_D + L_prior

train_op = tf.train.AdagradOptimizer(1e-1).minimize(loss)
#train_op = tf.train.AdamOptimizer(1e-2).minimize(loss)

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

# ---------- test data ------------- #
Xi, Xs, Xv = select_rows(X, np.arange(X.shape[0]))
Xindices   = np.arange(X.shape[0])
Yte_idx_comp, Yte_shape, Yte_idx_prot, Yte_val = select_y(Ytest, np.arange(Ytest.shape[0]))

# ------- train data (all) --------- #
Ytr_idx_comp, Ytr_shape, Ytr_idx_prot, Ytr_val = select_y(Ytrain, np.arange(Ytrain.shape[0]))

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

#with tf.Session() as sess:
sess = tf.Session()
if True:
  sess.run(tf.initialize_all_variables())

  for epoch in range(300):
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
                                    tb_ratio:   Ytrain.nnz / float(by_val.shape[0]),
                                    #beta.prec:  5.0 * np.ones( beta.shape[-1] ),
                                    V.prec:     5.0 * np.ones( V.shape[-1] ),
                                    Z.prec:     5.0 * np.ones( Z.shape[-1] )
                                    })

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
#beta.prec_div() + Z.prec_div() + V.prec_div() + beta.normal_div() + Z.normal_div() + V.normal_div()
      Ltr = sess.run([L_D, loss, beta.prec_div(), beta.normal_div()],
                     feed_dict={x_indices:  Xi,
                               x_shape:    Xs,
                               x_ids_val:  Xv,
                               x_idx_comp: Xindices,
                               y_idx_comp: Ytr_idx_comp,
                               y_idx_prot: Ytr_idx_prot,
                               y_val:      Ytr_val,
                               tb_ratio:   1.0,
                               #beta.prec:  5.0 * np.ones( beta.shape[-1] ),
                               V.prec:     5.0 * np.ones( V.shape[-1] ),
                               Z.prec:     5.0 * np.ones( Z.shape[-1] )
                               })
      beta_l2      = np.sqrt(sess.run(tf.nn.l2_loss(beta.mean)))
      beta_std_min = np.sqrt(sess.run(tf.reduce_min(beta.var)))
      beta_prec    = sess.run(beta.prec)
      V_prec       = sess.run(V.prec)
      Z_prec       = sess.run(Z.prec)
      #W2_l2 = sess.run(tf.nn.l2_loss(W2))
      test_rmse = np.sqrt( test_sse / Yte_val.shape[0])
      if epoch % 20 == 0:
          print("Epoch\tRMSE(test)\tL_D(train)\tloss(train)\tbeta divergence\t\tmin(beta.var)\trange(beta.prec)")
      print("%3d.\t%.5f\t\t%.2e\t%.2e\t[%.2e, %.2e]\t%.2e\t[%.1f, %.1f]" %
            (epoch, test_rmse, Ltr[0], Ltr[1], Ltr[2], Ltr[3], beta_std_min, beta_prec.min(), beta_prec.max()))



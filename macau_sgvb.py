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
h1_size     = 32
batch_size  = 512
reg         = 0.02
lrate0      = 0.08
lrate_decay = 1.0 #0.986
Y_prec      = 5.0

## inputs
y_val      = tf.placeholder(tf.float32)
y_idx_prot = tf.placeholder(tf.int64)
y_idx_comp = tf.placeholder(tf.int64)
x_indices  = tf.placeholder(tf.int64)
x_shape    = tf.placeholder(tf.int64)
x_ids_val  = tf.placeholder(tf.int64)
x_idx_comp = tf.placeholder(tf.int64) ## true compound indices

## ratio of total training points to mini-batch training points, for the current batch
tb_ratio = tf.placeholder(tf.float32)

## model
beta = vb.NormalGammaUni("beta", shape = [Nfeat, h1_size])
Z    = vb.NormalGammaUni("Z",    shape = [Ncomp, h1_size])
V    = vb.NormalGammaUni("V",    shape = [Nprot, h1_size])

## expected data log likelihood
sp_ids  = tf.SparseTensor(x_indices, x_ids_val, x_shape)

## means
Zmean_b = tf.nn.embedding_lookup(Z.mean, x_idx_comp)
h1      = tf.nn.embedding_lookup_sparse(beta.mean, sp_ids, None, combiner = "sum") + Zmean_b
h1_b    = tf.nn.embedding_lookup(h1, y_idx_comp)
Vmean_b = tf.nn.embedding_lookup(V.mean, y_idx_prot)
y_pred  = tf.squeeze(tf.batch_matmul(h1_b, Vmean_b, adj_y=True), [1, 2])
#y_pred = tf.squeeze(tf.batch_matmul(h1_b, Vmean_b, adj_y=True), [1, 2]) + tf.nn.embedding_lookup(b2, tf.squeeze(y_idx_prot, [1]))
y_loss  = Y_prec / 2.0 * tf.reduce_sum(tf.square(y_val - y_pred))

## variance
Zvar_b  = tf.nn.embedding_lookup(Z.var, x_idx_comp)
h1var   = tf.nn.embedding_lookup_sparse(beta.var, sp_ids, None, combiner = "sum") + Zvar_b
h1var_b = tf.nn.embedding_lookup(h1var, y_idx_comp)
Vvar_b  = tf.nn.embedding_lookup(V.var, y_idx_prot)

E_ysq   = tf.add(h1var_b, tf.square(h1_b))
y_var1  = Y_prec / 2.0 * tf.reduce_sum(tf.squeeze(tf.batch_matmul(E_ysq, Vvar_b, adj_y=True), [1, 2]))
y_var2  = Y_prec / 2.0 * tf.squeeze(tf.batch_matmul(h1var_b, tf.square(Vmean_b), adj_y=True), [1, 2])

L_D     = tb_ratio * (y_loss + y_var1 + y_var2)

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

#Xi, Xs, Xv = select_rows(X, np.arange(X.shape[0]))
X_ids      = np.arange(2)
Xi, Xs, Xv = select_rows(X, X_ids)
Yte_idx_comp, Yte_shape, Yte_idx_prot, Yte_val = select_y(Ytest, np.arange(Ytest.shape[0]))



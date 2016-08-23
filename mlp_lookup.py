import tensorflow as tf
import scipy.io
import numpy as np
import chemblnet as cn
from scipy.sparse import hstack

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--reg",   type=float, help="regularization for layers", default = 0.001)
parser.add_argument("--zreg",  type=float, help="regularization for Z (lookup table)", default = 0.001)
parser.add_argument("--hsize", type=int,   help="size of the hidden layer", default = 100)
args = parser.parse_args()

label = scipy.io.mmread("chembl-IC50-346targets.mm")
X     = scipy.io.mmread("chembl-IC50-compound-feat.mm").tocsr()

Ytrain, Ytest = cn.make_train_test(label, 0.2, seed = 123456)
Ytrain = Ytrain.tocsr()
Ytest  = Ytest.tocsr()

Nfeat  = X.shape[1]
Nprot  = Ytrain.shape[1]
Ncmpd  = Ytrain.shape[0]


batch_size = 100
h_size     = args.hsize
reg        = args.reg
zreg       = args.zreg
lrate      = 0.001
lrate_decay = 0.1 #0.986
lrate_min  = 3e-5
epsilon    = 1e-5

print("Num compounds:  %d" % Ncmpd)
print("Num proteins:   %d" % Nprot)
print("Num features:   %d" % Nfeat)
print("St. deviation:  %f" % np.std( Ytest.data ))
print("-----------------------")
print("Hidden size:    %d" % h_size)
print("reg:            %.1e" % reg)
print("Z-reg:          %.1e" % zreg)
print("Learning rate:  %.1e" % lrate)
print("-----------------------")

## variables for the model
W1 = tf.Variable(tf.random_uniform([Nfeat, h_size], minval=-1/np.sqrt(Nfeat), maxval=1/np.sqrt(Nfeat)))
b1 = tf.Variable(tf.random_uniform([h_size], minval=-1/np.sqrt(h_size), maxval=1/np.sqrt(h_size)))
W2 = tf.Variable(tf.random_uniform([Nprot, h_size], minval=-1/np.sqrt(h_size), maxval=1/np.sqrt(h_size)))
# b2 = tf.Variable(tf.constant(Ytrain.data.mean(), shape=[Nprot], dtype=tf.float32))
b2 = tf.Variable(tf.random_uniform([Nprot], minval=-1/np.sqrt(Nprot), maxval=1/np.sqrt(Nprot)))
b2g = tf.constant(Ytrain.data.mean(), dtype=tf.float32)
Z  = tf.Variable(tf.random_uniform([Ncmpd, h_size], minval=-1/np.sqrt(h_size), maxval=1/np.sqrt(h_size)))

## inputs
y_val      = tf.placeholder(tf.float32)
y_idx_prot = tf.placeholder(tf.int64)
y_idx_comp = tf.placeholder(tf.int64)
z_idx      = tf.placeholder(tf.int64)
sp_indices = tf.placeholder(tf.int64)
sp_shape   = tf.placeholder(tf.int64)
sp_ids_val = tf.placeholder(tf.int64)
tr_ind     = tf.placeholder(tf.bool)

def l1_reg(tensor, weight=1.0, scope=None):
  with tf.op_scope([tensor], scope, 'L1Regularizer'):
    l1_weight = tf.convert_to_tensor(weight,
                                     dtype=tensor.dtype.base_dtype,
                                     name='weight')
    return tf.mul(l1_weight, tf.reduce_sum(tf.abs(tensor)), name='value')

def batch_norm_wrapper(inputs, is_training, decay = 0.999):
    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

    if is_training is not None:
        batch_mean, batch_var = tf.nn.moments(inputs,[0])
        train_mean = tf.assign(pop_mean,
                               pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var,
                              pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs,
                train_mean, train_var, beta, scale, epsilon)
    else:
        return tf.nn.batch_normalization(inputs,
            pop_mean, pop_var, beta, scale, epsilon)

## regularization parameter
lambda_reg = tf.placeholder(tf.float32)
lambda_zreg = tf.placeholder(tf.float32)
learning_rate = tf.placeholder(tf.float32)

## model setup
sp_ids     = tf.SparseTensor(sp_indices, sp_ids_val, sp_shape)
# h1         = tf.nn.elu(tf.nn.embedding_lookup_sparse(W1, sp_ids, None, combiner = "sum") + b1)
# h1         = tf.nn.relu6(tf.nn.embedding_lookup_sparse(W1, sp_ids, None, combiner = "sum") + b1)
l1         = tf.nn.embedding_lookup_sparse(W1, sp_ids, None, combiner = "sum") + b1
Ze         = tf.nn.embedding_lookup(Z, z_idx)
h1         = tf.tanh(l1) + Ze

## batch normalization doesn't work that well in comparison to Torch 
# h1         = batch_norm_wrapper(l1, tr_ind)

h1e        = tf.nn.embedding_lookup(h1, y_idx_comp)
W2e        = tf.nn.embedding_lookup(W2, y_idx_prot)
b2e        = tf.nn.embedding_lookup(b2, tf.squeeze(y_idx_prot, [1]))
l2         = tf.squeeze(tf.batch_matmul(h1e, W2e, adj_y=True), [1, 2]) + b2e
y_pred     = l2 + b2g

## batch normalization doesn't work that well in comparison to Torch 
# scale2e    = tf.nn.embedding_lookup(scale2, tf.squeeze(y_idx_prot, [1]))
# beta2e     = tf.nn.embedding_lookup(beta2, tf.squeeze(y_idx_prot, [1]))
# batch_mean2, batch_var2 = tf.nn.moments(l2,[0])
# z2         = (l2 - batch_mean2) / tf.sqrt(batch_var2 + epsilon)
# y_pred     = scale2e * l2 + b2g

b_ratio = np.float32(Ncmpd) / np.float32(batch_size)

y_loss     = tf.reduce_sum(tf.square(y_val - y_pred))
#l2_reg     = lambda_reg * tf.global_norm((W1, W2))**2 + lambda_zreg * b_ratio * tf.nn.l2_loss(Ze)
l2_reg     = lambda_reg * tf.global_norm((W1, W2))**2 + lambda_zreg * tf.nn.l2_loss(Z)
loss       = l2_reg + y_loss/np.float32(batch_size)

# Use the adam optimizer
train_op   = tf.train.AdamOptimizer(learning_rate).minimize(loss)

def select_rows(X, row_idx):
  Xtmp = X[row_idx]
  indices = np.zeros((Xtmp.nnz, 1), dtype = np.int64)
  for i in range(row_idx.shape[0]):
    indices[ Xtmp.indptr[i] : Xtmp.indptr[i+1], 0 ] = i
  shape = [row_idx.shape[0], X.shape[1]]
  return indices, shape, Xtmp.indices.astype(np.int64, copy=False)

def select_y(X, row_idx):
  Xtmp = X[row_idx]
  indices = np.zeros((Xtmp.nnz, 1), dtype = np.int64)
  for i in range(row_idx.shape[0]):
    indices[ Xtmp.indptr[i] : Xtmp.indptr[i+1], 0 ] = i
  shape = [row_idx.shape[0], X.shape[1]]
  return indices, shape, Xtmp.indices.astype(np.int64, copy=False).reshape(-1, 1), Xtmp.data.astype(np.float32, copy=False)

Xi, Xs, Xv = select_rows(X, np.arange(X.shape[0]))
Yte_idx_comp, Yte_shape, Yte_idx_prot, Yte_val = select_y(Ytest, np.arange(Ytest.shape[0]))
Ytr_idx_comp, Ytr_shape, Ytr_idx_prot, Ytr_val = select_y(Ytrain, np.arange(Ytrain.shape[0]))

with tf.Session() as sess:
  sess.run(tf.initialize_all_variables())
  best_train_sse = np.inf
  decay_cnt = 0

  for epoch in range(200):
    rIdx = np.random.permutation(Ytrain.shape[0])
    
    if decay_cnt > 2:
      lrate = np.max( [lrate * lrate_decay, lrate_min] )
      decay_cnt = 0
      best_train_sse = train_sse
      if lrate <= 1e-6:
          print("Converged, stopping at learning rate of 1e-6.")
          break

    ## mini-batch loop
    for start in np.arange(0, Ytrain.shape[0], batch_size):
      idx = rIdx[start : min(Ytrain.shape[0], start + batch_size)]
      bx_indices, bx_shape, bx_ids_val           = select_rows(X, idx)
      by_idx_comp, by_shape, by_idx_prot, by_val = select_y(Ytrain, idx)

      sess.run(train_op, feed_dict={sp_indices: bx_indices,
                                    sp_shape:   bx_shape,
                                    sp_ids_val: bx_ids_val,
                                    z_idx:      idx,
                                    y_idx_comp: by_idx_comp,
                                    y_idx_prot: by_idx_prot,
                                    y_val:      by_val,
                                    tr_ind:     True,
                                    lambda_reg:  reg,
                                    lambda_zreg: zreg,
                                    learning_rate: lrate})


    ## epoch's Ytest error
    if epoch % 1 == 0:
      test_sse = sess.run(y_loss,  feed_dict = {sp_indices: Xi,
                                                 sp_shape:   Xs,
                                                 sp_ids_val: Xv,
                                                 z_idx:      np.arange(0, Ytest.shape[0]),
                                                 y_idx_comp: Yte_idx_comp,
                                                 y_idx_prot: Yte_idx_prot,
                                                 y_val:      Yte_val,
                                                 tr_ind:     False})
      train_sse = sess.run(y_loss, feed_dict = {sp_indices: Xi,
                                                 sp_shape:   Xs,
                                                 sp_ids_val: Xv,
                                                 z_idx:      np.arange(0, Ytrain.shape[0]),
                                                 y_idx_comp: Ytr_idx_comp,
                                                 y_idx_prot: Ytr_idx_prot,
                                                 y_val:      Ytr_val,
                                                 tr_ind:     False})
      if train_sse <= best_train_sse:
        best_train_sse = train_sse
      else:
        decay_cnt += 1

      W1_l2 = sess.run(tf.nn.l2_loss(W1))
      W2_l2 = sess.run(tf.nn.l2_loss(W2))
      Z_l2  = sess.run(tf.nn.l2_loss(Z))
      test_rmse = np.sqrt( test_sse / Yte_val.shape[0])
      train_rmse = np.sqrt( train_sse / Ytr_val.shape[0])

      print("%3d. RMSE(test) = %.5f   RMSE(train) = %.5f   ||W1|| = %.2f   ||W2|| = %.2f ||Z|| = %.2f  lr = %.0e" % (epoch, test_rmse, train_rmse, np.sqrt(W1_l2), np.sqrt(W2_l2), np.sqrt(Z_l2), lrate) )

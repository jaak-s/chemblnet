import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--reg",   type=float, help="regularization for layers", default = 1e-3)
parser.add_argument("--hsize", type=int,   help="size of the hidden layer", default = 100)
parser.add_argument("--side",  type=str,   help="side information", default = "chembl-IC50-compound-feat.mm")
parser.add_argument("--y",     type=str,   help="matrix", default = "chembl-IC50-346targets.mm")
parser.add_argument("--batch-size", type=int,   help="batch size", default = 100)
parser.add_argument("--epochs", type=int,  help="number of epochs", default = 100)
parser.add_argument("--model", type=str,
                    help = "Network model",
                    choices = ["mlp"],
                    default = "mlp")

args = parser.parse_args()

import tensorflow as tf
import scipy.io
import numpy as np
import chemblnet as cn

label = scipy.io.mmread(args.y)
X     = scipy.io.mmread(args.side).tocsr()

Ytrain, Ytest = cn.make_train_test(label, 0.2)
Ytrain = Ytrain.tocsr()
Ytest  = Ytest.tocsr()

Nfeat  = X.shape[1]
Nprot  = Ytrain.shape[1]
Ncmpd  = Ytrain.shape[0]

batch_size = args.batch_size
h_size     = args.hsize
reg        = args.reg
res_reg    = 3e-3
lrate      = 0.001
lrate_decay = 0.1 #0.986
lrate_min  = 3e-5
model      = args.model

print("Matrix:         %s" % args.y)
print("Side info:      %s" % args.side)
print("Num compounds:  %d" % Ncmpd)
print("Num proteins:   %d" % Nprot)
print("Num features:   %d" % Nfeat)
print("St. deviation:  %f" % np.std( Ytest.data ))
print("-----------------------")
print("Hidden size:    %d" % h_size)
print("reg:            %.1e" % reg)
print("Learning rate:  %.1e" % lrate)
print("Batch size:     %d"   % batch_size)
print("Model:          %s"   % model)
print("-----------------------")

## variables for the model
W1   = tf.Variable(tf.random_uniform([Nfeat, h_size], minval=-1/np.sqrt(Nfeat), maxval=1/np.sqrt(Nfeat)))
b1   = tf.Variable(tf.random_uniform([h_size], minval=-1/np.sqrt(h_size), maxval=1/np.sqrt(h_size)))

W1_5 = tf.Variable(tf.random_uniform([h_size, h_size], minval=-1/np.sqrt(h_size), maxval=1/np.sqrt(h_size)))
b1_5 = tf.Variable(tf.random_uniform([h_size], minval=-1/np.sqrt(h_size), maxval=1/np.sqrt(h_size)))

W2 = tf.Variable(tf.random_uniform([Nprot, h_size], minval=-1/np.sqrt(h_size), maxval=1/np.sqrt(h_size)))
# b2 = tf.Variable(tf.constant(Ytrain.data.mean(), shape=[Nprot], dtype=tf.float32))
b2 = tf.Variable(tf.random_uniform([Nprot], minval=-1/np.sqrt(Nprot), maxval=1/np.sqrt(Nprot)))
b2g = tf.constant(Ytrain.data.mean(), dtype=tf.float32)

## inputs
y_val      = tf.placeholder(tf.float32)
y_idx_prot = tf.placeholder(tf.int64)
y_idx_comp = tf.placeholder(tf.int64)
sp_indices = tf.placeholder(tf.int64)
sp_shape   = tf.placeholder(tf.int64)
sp_ids_val = tf.placeholder(tf.int64)
tr_ind     = tf.placeholder(tf.bool)

## regularization parameter
lambda_reg = tf.placeholder(tf.float32)
learning_rate = tf.placeholder(tf.float32)

## model setup
sp_ids     = tf.SparseTensor(sp_indices, sp_ids_val, sp_shape)
# h1         = tf.nn.elu(tf.nn.embedding_lookup_sparse(W1, sp_ids, None, combiner = "sum") + b1)
# h1         = tf.nn.relu6(tf.nn.embedding_lookup_sparse(W1, sp_ids, None, combiner = "sum") + b1)
l1         = tf.nn.embedding_lookup_sparse(W1, sp_ids, None, combiner = "sum") + b1
h1         = tf.tanh(l1)

## add another layer
h1_5       = tf.tanh( tf.nn.bias_add(tf.matmul(W1_5, h1), b1_5) )

## output layer
h1e        = tf.nn.embedding_lookup(h1_5, y_idx_comp)
W2e        = tf.nn.embedding_lookup(W2, y_idx_prot)
b2e        = tf.nn.embedding_lookup(b2, tf.squeeze(y_idx_prot, [1]))
l2         = tf.squeeze(tf.batch_matmul(h1e, W2e, adj_y=True), [1, 2]) + b2e
y_pred     = l2 + b2g

y_loss     = tf.reduce_sum(tf.square(y_val - y_pred))
l2_reg     = lambda_reg * tf.global_norm((W1, W1_5, W2))**2
loss       = l2_reg + y_loss/np.float32(batch_size)

# Use the adam optimizer
train_op   = tf.train.AdamOptimizer(learning_rate).minimize(loss)

def select_rows(X, row_idx):
  Xtmp = X[row_idx]
  indices = np.zeros((Xtmp.nnz, 1), dtype = np.int64)
## TODO: check if np.where is faster than this custom code
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

  for epoch in range(args.epochs):
    rIdx = np.random.permutation(Ytrain.shape[0])
    
    if decay_cnt > 2:
      lrate = lrate * lrate_decay
      decay_cnt = 0
      best_train_sse = train_sse
      if lrate <= 1e-6:
          print("Converged, stopping at learning rate of 1e-6.")
          break

    ## mini-batch loop (skipping the last batch)
    for start in np.arange(0, Ytrain.shape[0] - batch_size + 1, batch_size):
      idx = rIdx[start : min(Ytrain.shape[0], start + batch_size)]
      bx_indices, bx_shape, bx_ids_val           = select_rows(X, idx)
      by_idx_comp, by_shape, by_idx_prot, by_val = select_y(Ytrain, idx)

      sess.run(train_op, feed_dict={sp_indices: bx_indices,
                                    sp_shape:   bx_shape,
                                    sp_ids_val: bx_ids_val,
                                    y_idx_comp: by_idx_comp,
                                    y_idx_prot: by_idx_prot,
                                    y_val:      by_val,
                                    tr_ind:     True,
                                    lambda_reg: reg,
                                    learning_rate: lrate})


    ## epoch's Ytest error
    if epoch % 1 == 0:
      test_sse = sess.run(y_loss,  feed_dict = {sp_indices: Xi,
                                                 sp_shape:   Xs,
                                                 sp_ids_val: Xv,
                                                 y_idx_comp: Yte_idx_comp,
                                                 y_idx_prot: Yte_idx_prot,
                                                 y_val:      Yte_val,
                                                 tr_ind:     False})
      train_sse = sess.run(y_loss, feed_dict = {sp_indices: Xi,
                                                 sp_shape:   Xs,
                                                 sp_ids_val: Xv,
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
      test_rmse = np.sqrt( test_sse / Yte_val.shape[0])
      train_rmse = np.sqrt( train_sse / Ytr_val.shape[0])

      print("%3d. RMSE(test) = %.5f   RMSE(train) = %.5f   ||W1|| = %.5f   ||W2|| = %.5f   lr = %.0e" % (epoch, test_rmse, train_rmse, np.sqrt(W1_l2), np.sqrt(W2_l2), lrate) )

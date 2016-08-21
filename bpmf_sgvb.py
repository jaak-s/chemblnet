import tensorflow as tf
import scipy.io
import numpy as np
import chembl_data as cd
import vbutils as vb
import os

Ytrain = scipy.io.mmread(os.path.expanduser("~/Downloads/movielens-mm/train_df.mm"))
Ytest  = scipy.io.mmread(os.path.expanduser("~/Downloads/movielens-mm/test_df.mm"))

Ytrain = Ytrain.tocsr()
Ytest  = Ytest.tocsr()

Ncomp  = Ytrain.shape[0]
Nprot  = Ytrain.shape[1]
print("St. deviation (test):  %f" % np.std( Ytest.data ))

# learning parameters
Y_prec      = 2.0
h1_size     = 32
batch_size  = 256

extra_info  = False

## inputs
y_val      = tf.placeholder(tf.float32)
y_idx_prot = tf.placeholder(tf.int64)
y_idx_comp = tf.placeholder(tf.int64) ## true compound indices

learning_rate = tf.placeholder(tf.float32, name = "learning_rate")

## ratio of total training points to mini-batch training points, for the current batch
tb_ratio = tf.placeholder(tf.float32, name = "tb_ratio")

## model
Z     = vb.NormalGammaUni("Z", shape = [Ncomp, h1_size], initial_stdev = 0.1, fixed_prec = False)
V     = vb.NormalGammaUni("V", shape = [Nprot, h1_size], initial_stdev = 0.1, fixed_prec = False)
global_mean = tf.constant(Ytrain.data.mean(), dtype=tf.float32)

## means
Z_batch = tf.nn.embedding_lookup(Z.mean, y_idx_comp)
V_batch = tf.nn.embedding_lookup(V.mean, y_idx_prot)
y_pred  = global_mean + tf.squeeze(tf.batch_matmul(Z_batch, V_batch, adj_y=True), [1, 2])
y_loss  = Y_prec / 2.0 * tf.reduce_sum(tf.square(y_val - y_pred))

## variance
Zv_batch = tf.exp(tf.nn.embedding_lookup(Z.logvar, y_idx_comp))
Vv_batch = tf.exp(tf.nn.embedding_lookup(V.logvar, y_idx_prot))

E_Zsq   = tf.add(Zv_batch, tf.square(Z_batch))
y_var1  = Y_prec / 2.0 * tf.reduce_sum(tf.squeeze(tf.batch_matmul(E_Zsq, Vv_batch, adj_y=True), [1, 2]))
y_var2  = Y_prec / 2.0 * tf.reduce_sum(tf.squeeze(tf.batch_matmul(Zv_batch, tf.square(V_batch), adj_y=True), [1, 2]))

L_D     = tb_ratio * (y_loss + y_var1 + y_var2)
L_prior = Z.prec_div() + V.prec_div() + Z.normal_div() + V.normal_div()
loss    = L_D + L_prior

#train_op = tf.train.AdagradOptimizer(1e-1).minimize(loss)
train_op = tf.train.AdamOptimizer(1e-2).minimize(loss)
#train_op = tf.train.MomentumOptimizer(1e-7, 0.90).minimize(loss)

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
rIdx = np.random.permutation(Ytrain.shape[0])

# ---------- test data ------------- #
Yte_idx_comp, Yte_shape, Yte_idx_prot, Yte_val = select_y(Ytest, np.arange(Ytest.shape[0]))

# ------- train data (all) --------- #
Ytr_idx_comp, Ytr_shape, Ytr_idx_prot, Ytr_val = select_y(Ytrain, np.arange(Ytrain.shape[0]))

if False:
  sess = tf.Session()
  sess.run(tf.initialize_all_variables())
  test_sse = sess.run(tf.sqrt(tf.reduce_mean(tf.square(y_val - y_pred))),
                      feed_dict = {y_idx_comp: Yte_idx_comp,
                                   y_idx_prot: Yte_idx_prot,
                                   y_val:      Yte_val
                                   })

#with tf.Session() as sess:
sess = tf.Session()
if True:
  sess.run(tf.initialize_all_variables())

  for epoch in range(200):
    rIdx = np.random.permutation(Ytrain.shape[0])

    ## mini-batch loop
    for start in np.arange(0, Ytrain.shape[0], batch_size):
      if start + batch_size > Ytrain.shape[0]:
        break
      idx = rIdx[start : start + batch_size]
      by_idx_comp, by_shape, by_idx_prot, by_val = select_y(Ytrain, idx)

      sess.run(train_op, feed_dict={y_idx_prot: by_idx_prot,
                                    y_idx_comp: idx[by_idx_comp],
                                    y_val:      by_val,
                                    tb_ratio:   Ytrain.nnz / float(by_val.shape[0])
                                    })

    ## epoch's Ytest error
    if epoch % 1 == 0:
      test_rmse = sess.run(tf.sqrt(tf.reduce_mean(tf.square(y_val - y_pred))),
                           feed_dict = {y_idx_comp: Yte_idx_comp,
                                        y_idx_prot: Yte_idx_prot,
                                        y_val:      Yte_val
                                       })
      train_rmse = sess.run(tf.sqrt(tf.reduce_mean(tf.square(y_val - y_pred))),
                           feed_dict = {y_idx_comp: Ytr_idx_comp,
                                        y_idx_prot: Ytr_idx_prot,
                                        y_val:      Ytr_val
                                       })
      Ltr = sess.run([L_D, loss],
                     feed_dict={y_idx_comp: Ytr_idx_comp,
                                y_idx_prot: Ytr_idx_prot,
                                y_val:      Ytr_val,
                                tb_ratio:   1.0
                               })
      V_l2         = np.sqrt(sess.run(tf.nn.l2_loss(V.mean)))
      Z_l2         = np.sqrt(sess.run(tf.nn.l2_loss(Z.mean)))
      V_prec       = sess.run(V.prec)
      Z_prec       = sess.run(Z.prec)
      #W2_l2 = sess.run(tf.nn.l2_loss(W2))
      if epoch % 20 == 0:
          print("Epoch\tRMSE(te,tr)      | L_D,loss(train)\tl2: V.mu, Z.mu\tV.prec    | Z.prec")
      print("%3d.\t%.5f, %.5f | %.2e, %.2e\t%.2f, %.2f\t%.1f - %.1f | %.1f - %.1f" %
            (epoch, test_rmse, train_rmse, Ltr[0], Ltr[1], V_l2, Z_l2, V_prec.min(), V_prec.max(), Z_prec.min(), Z_prec.max() ))
      if extra_info:
          #print("beta: [%s]" % beta.summarize(sess))
          #print("Z:    [%s]" % Z.summarize(sess))
          print("V:    [%s]" % V.summarize(sess))


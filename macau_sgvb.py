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
noise_prec  = 5.0

## inputs
y_val      = tf.placeholder(tf.float32)
y_idx_prot = tf.placeholder(tf.int64)
y_idx_comp = tf.placeholder(tf.int64)
x_indices  = tf.placeholder(tf.int64)
x_shape    = tf.placeholder(tf.int64)
x_ids_val  = tf.placeholder(tf.int64)
x_idx_comp = tf.placeholder(tf.int64) ## true compound indices

## ratio of total training points to mini-batch training points
tb_ratio = tf.placeholder(tf.float32)

## model
beta = vb.NormalGammaUni("beta", shape = [Nfeat, h1_size])
Z    = vb.NormalGammaUni("Z",    shape = [Ncomp, h1_size])
V    = vb.NormalGammaUni("V",    shape = [Nprot, h1_size])

## expected data log likelihood
sp_id  = tf.SparseTensor(x_indices, x_ids_val, x_shape)
h1     = (tf.nn.embedding_lookup_sparse(beta.mean, sp_ids, None, combiner = "sum")
          + tf.nn.embedding_lookup(Z.mean, x_idx_comp) )

h1e    = tf.nn.embedding_lookup(h1, y_idx_comp)
Vmean_e = tf.nn.embedding_lookup(V.mean, y_idx_prot)
## currently bias term is commented out
y_pred = tf.squeeze(tf.batch_matmul(h1e, Vmean_e, adj_y=True), [1, 2]) #+ tf.nn.embedding_lookup(b2, tf.squeeze(y_idx_prot, [1]))
y_loss = noise_prec / 2.0 * tf.reduce_sum(tf.square(y_val - y_pred))

## variance
h1var  = (tf.nn.embedding_lookup_sparse(beta.var, sp_ids, None, combiner = "sum")
          + tf.nn.embedding_lookup(Z.var, x_idx_comp) )
h1vare = tf.nn.embedding_lookup(h1, y_idx_comp)
Vvare  = tf.nn.embedding_lookup(V.var, y_idx_prot)

E_ysq  = tf.add(h1vare, tf.square(h1e))
y_var1 = noise_prec / 2.0 * tf.reduce_sum(tf.squeeze(tf.batch_matmul(E_ysq, Vvare, adj_y=True), [1, 2]))
y_var2 = noise_prec / 2.0 * tf.squeeze(tf.batch_matmul(h1vare, tf.square(Vmean_e), adj_y=True), [1, 2])

L_D    = tb_ratio(y_loss + y_var1 + y_var2)



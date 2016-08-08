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
Nprot  = Ytrain.shape[1]

print("St. deviation:   %f" % np.std( Ytest.data ))

h1_size    = 32
batch_size = 512
reg        = 0.02
lrate0     = 0.08
lrate_decay = 1.0 #0.986

## some testing code
beta = vb.NormalGammaUni("beta", [100, 5])

sess = tf.Session()
sess.run(tf.initialize_all_variables())
sess.run(beta.prec_negdiv(0.1, 0.1))


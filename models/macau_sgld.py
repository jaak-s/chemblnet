import configargparse

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


p = configargparse.ArgParser()
p.register('type','bool',str2bool) # add type keyword to registries

p.add('-c', '--config', required=True, is_config_file=True, help='Config file path')
p.add("--side",  type=str, help="side information")
p.add("--y",     type=str, help="matrix")
p.add("--lambda_b", type=float, help="regularization for beta", default = 5.0)
p.add("--lambda_u", type=float, help="regularization for u",    default = 5.0)
p.add("--lambda_v", type=float, help="regularization for v",    default = 5.0)
p.add('--h_size',       required=True, help='List of hidden sizes (convolutional).',type=int)
p.add('--learning_rates', required=True, action='append', help='List of applied learning rates.', type=float)
p.add('--lr_durations',   required=True, action='append', help='List of durations for learning rates in epochs.', type=int)
p.add('--batch_size', required=True, help='Size of the minibatch.', type=int)
p.add('--test_ratio', required=True, help='Ratio of the testset.', type=float)
p.add("--temperature", required=False, type=float, help="Noise multiplier for SGLD (default 1.0)", default=1.0)
p.add("--burnin", required=False, type=int, help="Burn-in for SGLD (default 20)", default=20)
p.add("--adaptive_lambda", required=False, type="bool", help="Adaptive lambda for SGLD (default True)", default=False)
p.add("--alpha", required=True, type=float, help="Noise precisoin (default 5.0)")
p.add("--optimizer", required=True, type=str, help="Optimizer to use", choices = ["sgd", "sgld", "adam", "rmsprop"])
p.add("--rmsprop_centered", required=False, type="bool", help="Center RMSProp", default=False)
p.add("--rmsprop_momentum", required=False, type=float, help="Momentum for RMSProp", default=0.0)
p.add("--board", required=False, type=str, help="board directory", default=None)
p.add("--save",  required=False, type=str, help="filename to save the model to", default = None)
p.add("--save_rmse", required=False, type=str, help="filename to save RMSEs", default = None)
args = p.parse_args()


import tensorflow as tf
import scipy.io
import numpy as np
import chemblnet as cn
from scipy.sparse import hstack
import os

epochs = sum(args.lr_durations)
label = scipy.io.mmread(args.y)
X     = scipy.io.mmread(args.side).tocsr()

board = args.board
save_rmse = args.save_rmse

identifier, ext = os.path.splitext(os.path.basename(args.config))
if board is None:
    board = "boards/" + identifier
if save_rmse is None:
    save_rmse = identifier + "-results.csv"

Ytrain, Ytest = cn.make_train_test(label, args.test_ratio)
Ytrain = Ytrain.tocsr()
Ytest  = Ytest.tocsr()

Nfeat  = X.shape[1]
Nprot  = Ytrain.shape[1]
Ncmpd  = Ytrain.shape[0]

batch_size = args.batch_size
h_size     = args.h_size
lambda_u   = args.lambda_u
lambda_v   = args.lambda_v
lambda_b   = args.lambda_b
alpha      = args.alpha
#lrate_decay = 0.1 #0.986
#lrate_min  = 3e-6
epsilon    = 1e-5
temperature = args.temperature
burnin      = args.burnin
adaptive_lambda = args.adaptive_lambda
lambda_a0   = 0.1
lambda_b0   = 0.1
adaptive_start = 5

## variables for the model
init_std = 0.1
lr_path = np.repeat(args.learning_rates[0], args.lr_durations[0])
for i in range(1, len(args.learning_rates)):
    lr_path = np.concatenate([lr_path, np.repeat(args.learning_rates[i], args.lr_durations[i])])

Ytest_std  = np.std( Ytest.data ) if Ytest.nnz > 0 else np.nan

print("Matrix:         %s" % args.y)
print("Side info:      %s" % args.side)
print("Test ratio:     %.2f" % args.test_ratio)
print("Num y train:    %d" % Ytrain.nnz)
print("Num y test:     %d" % Ytest.nnz)
print("Num compounds:  %d" % Ncmpd)
print("Num proteins:   %d" % Nprot)
print("Num features:   %d" % Nfeat)
print("Test stdev:     %f" % Ytest_std)
print("-----------------------")
print("Num epochs:     %d" % epochs)
print("Hidden size:    %d" % h_size)
print("Lambda u        %.1e" % lambda_u)
print("Lambda v        %.1e" % lambda_v)
print("Lambda beta     %.1e" % lambda_b)
print("Learning rate:  %s"   % args.learning_rates)
print("Batch size:     %d"   % batch_size)
print("Optimizer:      %s"   % args.optimizer)
print("-----------------------")
if args.optimizer == "rmsprop":
    print("[ RMSProp options ]")
    print("centered:       %r" % args.rmsprop_centered)
    print("momentum:       %r" % args.rmsprop_momentum)
if args.optimizer == "sgld":
    print("[ SGLD options ]")
    print("Burn-in:        %d"   % burnin)
    print("Temperature:    %.1e" % temperature)
    print("Adaptive lambda: %r"   % adaptive_lambda)

beta = tf.Variable(tf.truncated_normal([Nfeat, h_size], stddev=init_std))
#b1 = tf.Variable(tf.truncated_normal([h_size], ))
V    = tf.Variable(tf.truncated_normal([Nprot, h_size], stddev=init_std))
U    = tf.Variable(tf.truncated_normal([Ncmpd, h_size], stddev=init_std))

## inputs
y_val      = tf.placeholder(tf.float32)
y_idx      = tf.placeholder(tf.int64, shape=[None, 2])
u_idx      = tf.placeholder(tf.int64)
x_indices  = tf.placeholder(tf.int64)
x_shape    = tf.placeholder(tf.int64)
x_ids_val  = tf.placeholder(tf.int64)

## regularization parameter
lambda_u_ph = tf.placeholder(tf.float32)
lambda_v_ph = tf.placeholder(tf.float32)
lambda_b_ph = tf.placeholder(tf.float32)

learning_rate = tf.placeholder(tf.float32)

## model setup
global_mean = tf.constant(Ytrain.data.mean(), dtype=np.float32)
x_ids = tf.SparseTensor(x_indices, x_ids_val, x_shape)
u_ids = tf.nn.embedding_lookup(U, u_idx)
uside = tf.nn.embedding_lookup_sparse(beta, x_ids, None, combiner = "sum") + u_ids

#uside_lkp = tf.nn.embedding_lookup(uside, y_idx_comp)
#vside_lkp = tf.nn.embedding_lookup(V,     y_idx_prot)
y_hat     = tf.matmul(uside, V, transpose_b=True)
y_hat_sel = tf.gather_nd(y_hat, y_idx)
y_hat_sel = tf.add(y_hat_sel, global_mean)

b_ratio   = tf.placeholder(tf.float32)
y_sse     = 2 * tf.nn.l2_loss(y_val - y_hat_sel)
y_rmse    = tf.sqrt(tf.div(y_sse, tf.cast(tf.shape(y_val)[0], tf.float32)))

y_loss    = alpha * b_ratio * 0.5 * y_sse

prior     = lambda_b_ph * tf.nn.l2_loss(beta) + \
            lambda_u_ph * b_ratio * tf.nn.l2_loss(u_ids) + \
            lambda_v_ph * tf.nn.l2_loss(V)

loss      = prior + y_loss

# Use the adam optimizer
if args.optimizer == "adam":
    train_op   = tf.train.AdamOptimizer(learning_rate).minimize(loss)
elif args.optimizer == "sgd":
    train_op   = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
elif args.optimizer == "rmsprop":
    train_op   = tf.train.RMSPropOptimizer(learning_rate, centered=args.rmsprop_centered, momentum=args.rmsprop_momentum).minimize(loss)
elif args.optimizer == "sgld":
    train_op   = cn.SGLD(learning_rate, temp = temperature).minimize(loss)
    test_y_avg = cn.PosteriorMean()
    train_y_avg = cn.PosteriorMean()


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

def rmse(ytrue, yhat):
    return np.sqrt(np.mean(np.square(ytrue - yhat)))

def sample_lambda(theta, a0, b0):
    """ Samples lambda for theta (single scalar)."""
    lambda_a = a0 + np.prod(theta.shape) / 2.0
    lambda_b = b0 + np.square(theta).sum() / 2.0
    return np.random.gamma(lambda_a, 1.0 / lambda_b)

Xi, Xs, Xv = select_rows(X, np.arange(X.shape[0]))
Yte_idx_comp, Yte_shape, Yte_idx_prot, Yte_val = select_y(Ytest, np.arange(Ytest.shape[0]))
Ytr_idx_comp, Ytr_shape, Ytr_idx_prot, Ytr_val = select_y(Ytrain, np.arange(Ytrain.shape[0]))

Yte_idx = np.column_stack([Yte_idx_comp, Yte_idx_prot])
Ytr_idx = np.column_stack([Ytr_idx_comp, Ytr_idx_prot])

fd_test = dict()
fd_test[x_indices] = Xi
fd_test[x_shape]   = Xs
fd_test[x_ids_val] = Xv
fd_test[u_idx]     = np.arange(0, Ytest.shape[0])
fd_test[y_idx]     = Yte_idx
fd_test[y_val]     = Yte_val

fd_train = dict()
fd_train[x_indices] = Xi
fd_train[x_shape]   = Xs
fd_train[x_ids_val] = Xv
fd_train[u_idx]     = np.arange(0, Ytrain.shape[0])
fd_train[y_idx]     = Ytr_idx
fd_train[y_val]     = Ytr_val

if args.optimizer != "sgld":
    header = "     RMSE(tr)  RMSE(te)\t| lr"
else:
    header = "     RMSE(tr)  RMSE(te)\t| lmb_u\tlmb_v\tlmb_b\t| lr"

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  best_train_rmse = np.inf
  decay_cnt = 0

  test_rmse  = sess.run(y_rmse, feed_dict = fd_test)
  train_rmse = sess.run(y_rmse, feed_dict = fd_train)
  print("Test rmse before start:  %.5f" % test_rmse)
  print("Train rmse before start: %.5f" % train_rmse)

  for epoch in range(epochs):
    lrate = lr_path[epoch]
    rIdx = np.random.permutation(Ytrain.shape[0])
    
    #if decay_cnt > 5 and lrate >= lrate_min:
    #  lrate = np.max( [lrate * lrate_decay, lrate_min] )
    #  decay_cnt = 0
    #  best_train_rmse = train_rmse

    ## mini-batch loop
    for start in np.arange(0, Ytrain.shape[0] - batch_size + 1, batch_size):
      idx = rIdx[start : start + batch_size]
      bx_indices, bx_shape, bx_ids_val           = select_rows(X, idx)
      by_idx_comp, by_shape, by_idx_prot, by_val = select_y(Ytrain, idx)

      sess.run(train_op, feed_dict={x_indices: bx_indices,
                                    x_shape:   bx_shape,
                                    x_ids_val: bx_ids_val,
                                    u_idx:     idx,
                                    y_idx:     np.column_stack([by_idx_comp, by_idx_prot]),
                                    y_val:     by_val,
                                    b_ratio:   np.float32(Ytrain.shape[0]) / len(idx),
                                    lambda_u_ph: lambda_u,
                                    lambda_v_ph: lambda_v,
                                    lambda_b_ph: lambda_b,
                                    learning_rate: lrate})


    ## epoch's Ytest error
    if epoch % 20 == 0:
        print(header)

    if epoch % 1 == 0:
        if args.optimizer != "sgld":
            test_rmse  = sess.run(y_rmse, feed_dict = fd_test)
            train_rmse = sess.run(y_rmse, feed_dict = fd_train)
        else:
            ## SGLD
            if epoch == burnin:
                print("  ----- Burn-in complete, averaging samples -----")
            test_y_hat  = sess.run(y_hat_sel, feed_dict = fd_test)
            train_y_hat = sess.run(y_hat_sel, feed_dict = fd_train)

            test_y_avg.addSample(test_y_hat, epoch >= burnin)
            train_y_avg.addSample(train_y_hat, epoch >= burnin)

            test_rmse  = rmse(fd_test[y_val], test_y_avg.getMean())
            train_rmse = rmse(fd_train[y_val], train_y_avg.getMean())
            
            if adaptive_lambda and epoch >= adaptive_start:
                U_data, V_data, beta_data = sess.run([U, V, beta])
                lambda_u = sample_lambda(U_data, lambda_a0, lambda_b0)
                lambda_v = sample_lambda(V_data, lambda_a0, lambda_b0)
                lambda_b = sample_lambda(beta_data, lambda_a0, lambda_b0)

            #test_rmse1  = rmse(fd_test[y_val], test_y_hat)
            #train_rmse1 = rmse(fd_train[y_val], train_y_hat)
 

        if train_rmse <= best_train_rmse:
            best_train_rmse = train_rmse
        else:
            decay_cnt += 1

        if args.optimizer == "sgld":
            print("%3d. %.5f   %.5f\t| %.1f\t%.1f\t%.1f\t| %.1e" % (epoch, train_rmse, test_rmse, lambda_u, lambda_v, lambda_b, lrate) )
        else:
            print("%3d. %.5f   %.5f\t|  %.1e" % (epoch, train_rmse, test_rmse, lrate) )

  if save_rmse is not None:
      ## saving RMSE values
      if not os.path.isfile(save_rmse):
          with open(save_rmse, "w") as myfile:
              myfile.write("train_rmse,test_rmse\n")
      with open(save_rmse, "a") as myfile:
          myfile.write("%.6f,%.6f\n" % (train_rmse, test_rmse))

  ## after the training loop
  if args.save is not None:
    saver = tf.train.Saver()
    saver.save(sess, args.save)
    print("Saved model to '%s'." % args.save)


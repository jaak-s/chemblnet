import numpy as np
import scipy as sp
import numbers

def csr2indices(csr):
  counts  = (csr.indptr[1:] - csr.indptr[0:-1]).reshape(-1, 1)
  indices = np.zeros((csr.nnz, 1), dtype = np.int64)
  for i in range(csr.shape[0]):
    indices[ csr.indptr[i] : csr.indptr[i+1], 0 ] = i
  shape   = [0, 0]
  return indices, shape, csr.indices.astype(np.int64, copy=False)

class Data:
  def __init__(self, Xtr, Xte, Ytr, Yte):
    self.Xtrain = Xtr.tocsr()
    self.Ytrain = Ytr.reshape(-1, 1)
    self.Xtest  = Xte.tocsr()
    self.Ytest  = Yte.reshape(-1, 1)
    self.Nfeat  = Xtr.shape[1]

  def randomize_train(self):
    idx = range(len(self.Ytrain))
    np.random.shuffle(idx)
    self.Ytrain = self.Ytrain[idx]
    self.Xtrain = self.Xtrain[idx]

  def get_train_batch(self, start, batch_size):
    indices, shape, ids_val = csr2indices(self.Xtrain[start : start + batch_size])
    return indices, shape, ids_val, self.Ytrain[start : start + batch_size]


def make_target_col(data, label, col, ratio):
  idx  = (label.col == col).nonzero()[0]
  Y    = label.data[idx]
  rows = label.row[idx]
  X    = data.tocsr()[rows,:]

  ntrain = np.int(np.round((1 - ratio) * Y.shape[0]))
  rIdx  = np.random.permutation(Y.shape[0])
  trIdx = rIdx[0:ntrain]
  teIdx = rIdx[ntrain:]

  Xtr  = X[trIdx,:]
  Xte  = X[teIdx, :]
  Ytr  = Y[trIdx]
  Ymean = Ytr.mean()
  Ytr  = Ytr - Ymean
  Yte  = Y[teIdx] - Ymean

  return Xtr, Ytr, Xte, Yte

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

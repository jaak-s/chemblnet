import numpy as np

def csr2indices(csr):
  counts  = (csr.indptr[1:] - csr.indptr[0:-1]).reshape(-1, 1)
  indices = np.zeros((csr.nnz, 1), dtype = np.int64)
  for i in range(csr.shape[0]):
    indices[ csr.indptr[i] : csr.indptr[i+1], 0 ] = i
  shape   = [0, 0]
  return indices, shape, csr.indices.astype(np.int64, copy=False)

class Data:
  def __init__(self, Xtr, Xte, Ytr, Yte):
    self.Xtrain = Xtr
    self.Ytrain = Ytr.reshape(-1, 1)
    self.Xtest  = Xte
    self.Ytest  = Yte.reshape(-1, 1)
    self.Nfeat  = Xtr.shape[1]

def split_train_test(X, Y, ratio):
    nrow  = X.shape[0]
    ntest = int(round(nrow * ratio))
    rperm = np.random.permutation(nrow)
    train = rperm[ntest:]
    test  = rperm[0:ntest]
    return Data(X[train], X[test], Y[train], Y[test])


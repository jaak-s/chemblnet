import unittest
from chemblnet import PosteriorMean
import numpy as np

class TestPosteriorMean(unittest.TestCase):
    def test_pmean(self):
        X = np.random.randn(10, 2, 3)
        pm = PosteriorMean()
        pm.addSample(X[0,:], average = False)
        pm.addSample(X[1,:], average = False)
        pm.addSample(X[2,:], average = True)
        pm.addSample(X[3,:], average = True)
        self.assertTrue(np.allclose(pm.sample_avg, X[2:4, :].mean(0)))
        self.assertTrue(pm.n == 2)

        for i in range(4, X.shape[0]):
            pm.addSample(X[i,:], average = True)
        self.assertTrue(np.allclose(pm.sample_avg, X[2:, :].mean(0)))
        self.assertTrue(pm.n == 8)

        Xvar      = pm.getVar()
        Xsub      = X[2:, :]
        Xvar_true = np.square((Xsub - Xsub.mean(0))).sum(0) / (Xsub.shape[0] - 1)
        self.assertTrue(np.allclose(Xvar, Xvar_true))


if __name__ == '__main__':
    unittest.main()


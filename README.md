# ChEMBL Net

Python tools for building ChEMBL network.

# Installation
```
python setup.py install --user
```

You also need to install TensorFlow for running the models in this package.

## Installation in Conda

```
conda create --name chemblnet python=3
source activate chemblnet
conda install numpy scipy matplotlib
conda install ipython
conda install -c conda-forge tensorflow
```

# Model training example
First you need to download IC50 data and ECFP features:
```
wget http://homes.esat.kuleuven.be/~jsimm/chembl-IC50-346targets.mm
wget http://homes.esat.kuleuven.be/~jsimm/chembl-IC50-compound-feat.mm
```

Model directory contains several models, which you can then easily train:
```
python models/deep_macau.py --reg 1e-3 --zreg 1e-4 --hsize 100 \
  --model non_linear_z \
  --y chembl-IC50-346targets.mm \
  --side chembl-IC50-compound-feat.mm
```

Another example is `models/mlp_multi3.py` which uses multi-task MLP (multi-layer perceptron) with 1 hidden layer.

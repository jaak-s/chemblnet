from setuptools import setup

execfile('chemblnet/version.py')

setup(name='chemblnet',
      version='0.2',
      description='Neural Networks for ChEMBL',
      url='http://github.com/jaak-s/chemblnet',
      author='Jaak Simm',
      author_email='jaak.simm@gmail.com',
      license='MIT',
      packages=['chemblnet'],
      zip_safe=False)


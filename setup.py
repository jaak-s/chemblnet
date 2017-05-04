from setuptools import setup

exec(open('chemblnet/version.py').read())

setup(name='chemblnet',
      version=__version__,
      description='Neural Networks for ChEMBL',
      url='http://github.com/jaak-s/chemblnet',
      author='Jaak Simm',
      author_email='jaak.simm@gmail.com',
      license='MIT',
      packages=['chemblnet'],
      zip_safe=False)


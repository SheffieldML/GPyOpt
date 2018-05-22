#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from setuptools import setup, find_packages

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

__version__ = "1.2.5"

packages = find_packages(exclude=("GPyOpt.testing",))
setup(name = 'GPyOpt',
      version = __version__,
      author = read('AUTHORS.txt'),
      author_email = "j.h.gonzalez@sheffield.ac.uk",
      description = ("The Bayesian Optimization Toolbox"),
      license = "BSD 3-clause",
      keywords = "machine-learning gaussian-processes kernels optimization",
      url = "http://sheffieldml.github.io/GPyOpt/",
      packages = packages,
      package_dir = {'GPyOpt': 'GPyOpt'},
      include_package_data = True,
      py_modules = ['GPyOpt.__init__'],
      long_description = read('README.md'),
      install_requires = ['numpy>=1.7', 'scipy>=0.16', 'GPy>=1.8'],
      extras_require = {'optimizer':['DIRECT','cma','pyDOE','sobol_seq','emcee'],'docs':['matplotlib >=1.3','Sphinx','IPython']},
      classifiers=['License :: OSI Approved :: BSD License',
                   'Natural Language :: English',
                   'Operating System :: MacOS :: MacOS X',
                   'Operating System :: Microsoft :: Windows',
                   'Operating System :: POSIX :: Linux',
                   'Programming Language :: Python :: 2.7',
                   'Topic :: Scientific/Engineering :: Artificial Intelligence'],
      scripts=['gpyopt.py'],
     )

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from setuptools import setup

# Version number
version = '0.1.4'

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(name = 'GPyOpt',
      version = version,
      author = read('AUTHORS.txt'),
      author_email = "j.h.gonzalez@sheffield.ac.uk",
      description = ("The Bayesian Optimization Toolbox"),
      license = "BSD 3-clause",
      keywords = "machine-learning gaussian-processes kernels optimization",
      url = "http://sheffieldml.github.com/GPyOpt/",
      packages = ["GPyOpt.fmodels",
                  "GPyOpt.core",
                  "GPyOpt.methods",
                  "GPyOpt.plotting",
                  "GPyOpt.interface",
                  "GPyOpt.demos",
                  "GPyOpt.testing",
                  "GPyOpt.util"],
      package_dir={'GPyOpt': 'GPyOpt'},
      include_package_data = True,
      py_modules = ['GPyOpt.__init__'],
      long_description=read('README.md'),
      install_requires=['numpy>=1.7', 'scipy>=0.12', 'GPy>=0.6'],
      extras_require = {'optimizer':['DIRECT','cma','pyDOE'],'docs':['matplotlib >=1.3','Sphinx','IPython']},
      classifiers=['License :: OSI Approved :: BSD License',
                   'Natural Language :: English',
                   'Operating System :: MacOS :: MacOS X',
                   'Operating System :: Microsoft :: Windows',
                   'Operating System :: POSIX :: Linux',
                   'Programming Language :: Python :: 2.7',
                   'Topic :: Scientific/Engineering :: Artificial Intelligence'],
       scripts=['gpyopt.py'],
      )

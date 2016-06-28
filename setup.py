#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from setuptools import setup
from GPyOpt.__version__ import __version__

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(name = 'GPyOpt',
      version = __version__,
      author = read('AUTHORS.txt'),
      author_email = "j.h.gonzalez@sheffield.ac.uk",
      description = ("The Bayesian Optimization Toolbox"),
      license = "BSD 3-clause",
      keywords = "machine-learning gaussian-processes kernels optimization",
      url = "http://sheffieldml.github.io/GPyOpt/",
      packages = ["GPyOpt",
                  "GPyOpt.acquisitions",
                  "GPyOpt.core",
                  "GPyOpt.core.task",
                  "GPyOpt.core.evaluators",
                  "GPyOpt.interface",
                  "GPyOpt.methods",
                  "GPyOpt.models",
                  "GPyOpt.objective_examples",
                  "GPyOpt.optimization",
                  "GPyOpt.plotting",
                  "GPyOpt.util"],
      package_dir={'GPyOpt': 'GPyOpt'},
      include_package_data = True,
      py_modules = ['GPyOpt.__init__'],
      long_description=read('README.md'),
      install_requires=['numpy>=1.7', 'scipy>=0.16', 'GPy>=0.6'],
      extras_require = {'optimizer':['DIRECT','cma','pyDOE'],'docs':['matplotlib >=1.3','Sphinx','IPython'],'others':['pandas']},
      classifiers=['License :: OSI Approved :: BSD License',
                   'Natural Language :: English',
                   'Operating System :: MacOS :: MacOS X',
                   'Operating System :: Microsoft :: Windows',
                   'Operating System :: POSIX :: Linux',
                   'Programming Language :: Python :: 2.7',
                   'Topic :: Scientific/Engineering :: Artificial Intelligence'],
       scripts=['gpyopt.py'],
      )

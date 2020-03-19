GPyOpt
======

Gaussian process optimization using [GPy](http://sheffieldml.github.io/GPy/). Performs global optimization with different acquisition functions. Among other functionalities, it is possible to use GPyOpt to optimize physical experiments (sequentially or in batches) and tune the parameters of Machine Learning algorithms. It is able to handle large data sets via sparse Gaussian process models.

* [GPyOpt homepage](http://sheffieldml.github.io/GPyOpt/)
* [Tutorial Notebooks](http://nbviewer.ipython.org/github/SheffieldML/GPyOpt/blob/master/manual/index.ipynb)
* [Online documentation](http://gpyopt.readthedocs.io/)

[![licence](https://img.shields.io/badge/licence-BSD-blue.svg)](http://opensource.org/licenses/BSD-3-Clause)  [![develstat](https://travis-ci.org/SheffieldML/GPyOpt.svg?branch=master)](https://travis-ci.org/SheffieldML/GPyOpt) [![covdevel](http://codecov.io/github/SheffieldML/GPyOpt/coverage.svg?branch=master)](http://codecov.io/github/SheffieldML/GPyOpt?branch=master) [![Research software impact](http://depsy.org/api/package/pypi/GPyOpt/badge.svg)](http://depsy.org/package/python/GPyOpt)


### Citation

    @Misc{gpyopt2016,
      author =   {The GPyOpt authors},
      title =    {{GPyOpt}: A Bayesian Optimization framework in python},
      howpublished = {\url{http://github.com/SheffieldML/GPyOpt}},
      year = {2016}
    }

Getting started
===============

Installing with pip
-------------------
The simplest way to install GPyOpt is using pip. ubuntu users can do:

```bash
    sudo apt-get install python-pip
    pip install gpyopt
```

If you'd like to install from source, or want to contribute to the project (e.g. by sending pull requests via github), read on. Clone the repository in GitHub and add it to your $PYTHONPATH.

```bash
    git clone https://github.com/SheffieldML/GPyOpt.git
    cd GPyOpt
    python setup.py develop
```

Dependencies:
------------------------
  - GPy
  - paramz
  - numpy
  - scipy
  - matplotlib
  - DIRECT (optional)
  - cma (optional)
  - pyDOE (optional)
  - sobol_seq (optional)

You can install dependencies by running:
```
pip install -r requirements.txt
```


Funding Acknowledgements
========================
* [BBSRC Project No BB/K011197/1](http://staffwww.dcs.shef.ac.uk/people/N.Lawrence/projects/recombinant/) "Linking recombinant gene sequence to protein product manufacturability using CHO cell genomic resources"

* See GPy funding Acknowledgements







 # End of maintenance for GPyOpt

Dear GPyOpt community!

We would like to acknowledge the obvious. The core team of GPyOpt has moved on, and over the past months we weren't giving the package nearly as much attention as it deserves. Instead of dragging our feet and giving people only occasional replies and no new features, we feel the time has come to officially declare the end of GPyOpt maintenance and archive this repository.

We would like to thank the community that has formed around GPyOpt. Without your interest, discussions, bug fixes and pull requests the package would never be as successful as it is. We hope we were able to provide you with a useful tool to aid your research and work.

If you feel really enthusiastic and would like to take over the package, feel free to drop us an email, and who knows, maybe you'll be the one(s) carrying the GPyOpt to new heights!

Sincerely yours,
[Andrei Paleyes](https://paleyes.info/) and [Javier Gonzalez](https://javiergonzalezh.github.io/)


# GPyOpt

Gaussian process optimization using [GPy](http://sheffieldml.github.io/GPy/). Performs global optimization with different acquisition functions. Among other functionalities, it is possible to use GPyOpt to optimize physical experiments (sequentially or in batches) and tune the parameters of Machine Learning algorithms. It is able to handle large data sets via sparse Gaussian process models.

* [GPyOpt homepage](http://sheffieldml.github.io/GPyOpt/)
* [Tutorial Notebooks](http://nbviewer.ipython.org/github/SheffieldML/GPyOpt/blob/master/manual/index.ipynb)
* [Online documentation](http://gpyopt.readthedocs.io/)

[![licence](https://img.shields.io/badge/licence-BSD-blue.svg)](http://opensource.org/licenses/BSD-3-Clause)  [![develstat](https://travis-ci.org/SheffieldML/GPyOpt.svg?branch=master)](https://travis-ci.org/SheffieldML/GPyOpt) [![covdevel](http://codecov.io/github/SheffieldML/GPyOpt/coverage.svg?branch=master)](http://codecov.io/github/SheffieldML/GPyOpt?branch=master) [![Research software impact](http://depsy.org/api/package/pypi/GPyOpt/badge.svg)](http://depsy.org/package/python/GPyOpt)

### Citation

```
@Misc{gpyopt2016,
author = {The GPyOpt authors},
title = {{GPyOpt}: A Bayesian Optimization framework in python},
howpublished = {\url{http://github.com/SheffieldML/GPyOpt}},
year = {2016}
}
```

## Getting started

### Installing with pip

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

## Dependencies:

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


##  Funding Acknowledgements

* [BBSRC Project No BB/K011197/1](http://staffwww.dcs.shef.ac.uk/people/N.Lawrence/projects/recombinant/) "Linking recombinant gene sequence to protein product manufacturability using CHO cell genomic resources"

* See GPy funding Acknowledgements

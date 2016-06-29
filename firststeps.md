---
layout: page
title: Installation
subtitle: GPyOpt installation manual for users and developers
---


## General users: Installation using pip


GPyOpt (and GPy) requires the newest version (0.16) of scipy. We strongly recommend using the anaconda Python distribution. With anaconda you can update scipy and install GPyOpt is using [pip](https://pip.pypa.io/en/stable/installing/). Ubuntu users can do:

```
$ conda update scipy
$ pip install gpyopt
```

We have also been successful installing GPyOpt in OS and Windows machines. If you'd like to install from source, or want to contribute to the project (i.e. by sending pull requests via GitHub), read on.


## Solving installation problems

In you have problems installing GPyOpt with pip try to install it from source doing:

```
$ git clone https://github.com/SheffieldML/GPyOpt.git
$ cd GPyOpt
$ git checkout devel
$ nosetests GPyOpt/testing
```

## Dependencies

There are a number of dependencies that you need to install (using pip). Three of them are needed to ensure the good behaviour of the package:

- GPy (>=1.0.8)
- numpy (>=1.7)
- scipy (>=0.16)

Other dependencies are optional. All of them are also pip installable and include, pyDOE, cma, direct, scikit-learn and pandas.

## Ubuntu hackers

Most developers are using Ubuntu. To install the required packages:

```
$ sudo apt-get install python-numpy python-scipy python-matplotlib
```

To procced just clone this git repository and add it to your path:

```
$ git clone git@github.com:SheffieldML/GPyOpt.git ~/SheffieldML
$ echo 'PYTHONPATH=$PYTHONPATH:~/SheffieldML' >> ~/.bashrc
```
If you want to incorporate your changes to the repo, make a pull request but be sure that your code passes all the unit test.

## Running unit tests:

If you want send a pull request, please ensure that your changes pass the unittest. First, ensure that nose is installed. Otherwise do

```
$ pip install nose
```
and run the nosetest from the root directory of the repository.

```
$ nosetests -v GPyOpt/testing
```





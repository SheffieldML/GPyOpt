#!/usr/bin/env python
import matplotlib
matplotlib.use('agg')

import nose, warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

if __name__=='__main__': # for parallel computation
    nose.main('GPyOpt', defaultTest='GPyOpt/testing/', argv=[''])
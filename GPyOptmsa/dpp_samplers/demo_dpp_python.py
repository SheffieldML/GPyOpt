import os
os.chdir("/home/javier/workspace/GitHub/GPyOpt/GPyOpt/util/dpp_samplers")

import GPy
import matplotlib.pyplot as plt
from GPyOpt.util.general import multigrid
from dpp import *

# Genetate grid
Ngrid = 50
bounds = [(-2,2),(-2,2)]
X = multigrid(bounds, Ngrid)  

# Define kernel and kernel matrix
kernel = GPy.kern.RBF(len(bounds), variance=1, lengthscale=0.5) 
L = kernel.K(X)

# Number of points of each DPP sample
k = 50

# Putative inputs
set = [25,900, 1655,2125]

# Samples and plot from original and conditional with standard DPPS
sample          = sample_dpp(L,k)
sample_condset  = sample_conditional_dpp(L,set,k)

plt.subplot(1, 2, 1)
plt.plot(X[sample,0],X[sample,1],'.',)
plt.title('Sample from the DPP')
plt.subplot(1, 2, 2)
plt.plot(X[set,0],X[set,1],'k.',markersize=20)
plt.plot(X[sample_condset,0],X[sample_condset,1],'.',)
plt.title('Conditional sample from the DPP')


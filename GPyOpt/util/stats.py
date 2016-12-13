# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from ..util.general import samples_multidimensional_uniform
import numpy as np


def initial_design(design,space,data_init):

    if design == 'latin':
        samples = sample_initial_design(design,space,data_init)
        if space.has_constrains() == True:
            print('Sampling with constrains is now allowed with Latin designs.')

    elif space.has_constrains() == False:
        samples = sample_initial_design(design,space,data_init)

    elif design == 'random'  and space.has_constrains() == True:
        samples = np.empty((0,space.dimensionality))
        while samples.shape[0] < data_init:
            domain_samples = sample_initial_design(design, space, data_init)
            valid_indices = (space.indicator_constraints(domain_samples)==1).flatten()
            if sum(valid_indices)>0:
                valid_samples = domain_samples[valid_indices,:]
                samples = np.vstack((samples,valid_samples))
    return samples[0:data_init,:]


def sample_initial_design(design,space,data_init):
    """
    :param design: the choice of designs
    :param bounds: the boundary of initial points
    :param data_init: the number of initial points
    :Note: discrete dimensions are always added based on uniform samples
    """
    samples = np.empty((data_init,0))

    for d in space.space:
        for k in range(d['dimensionality']):
            if (d['type'] == 'continuous'):
                bounds = [(d['domain'][0],d['domain'][1])]
                sample_var = samples_multidimensional_uniform(bounds, data_init)

            if d['type'] == 'context':
                ### --- context is sampled in the interval
                # bounds = [(d['domain'][0],d['domain'][1])]
                # sample_var = samples_multidimensional_uniform(bounds, data_init)

                ### --- context is fixed
                sample_var = np.ones((data_init,1))*d['value']

            if (d['type'] == 'discrete') or (d['type'] == 'categorical') :
                sample_var = np.atleast_2d(np.random.choice(d['domain'],data_init)).T

            if (d['type'] == 'bandit'):
                sample_var = d['domain'][np.random.randint(d['domain'].shape[0],size=data_init),:]

            samples = np.hstack((samples,sample_var))

    if design == 'latin':
        index_nondiscrete = []
        bounds_nondiscrete = []
        i=0

        from pyDOE import lhs
        for d in space.space:
            for k in range(d['dimensionality']):
                if (d['type'] == 'continuous') or (d['type'] == 'context') :
                    bounds_nondiscrete += bounds
                    index_nondiscrete += [i]
                    i+=1

        X_design_aux = lhs(len(bounds_nondiscrete),data_init, criterion='center')
        lB = np.asarray(bounds)[:,0].reshape(1,len(bounds))
        uB = np.asarray(bounds)[:,1].reshape(1,len(bounds))
        diff = uB-lB
        I = np.ones((X_design_aux.shape[0],1))
        X_design = np.dot(I,lB) + X_design_aux*np.dot(I,diff)
        samples[:,index_nondiscrete] = X_design

    return samples

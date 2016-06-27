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

    if space.has_types['bandit']:
        arms = space.get_bandit()
        X_design = arms[np.random.randint(arms.shape[0],size=data_init),:]

    else:
        bounds = space.get_bounds()
        if design == 'random':
            X_design = samples_multidimensional_uniform(bounds, data_init)
        elif design == 'latin':
            try:
                from pyDOE import lhs
                # Generate points in hypercube
                X_design_aux = lhs(len(bounds),data_init, criterion='center')
                # Normalize to the give box constrains
                lB = np.asarray(bounds)[:,0].reshape(1,len(bounds))
                uB = np.asarray(bounds)[:,1].reshape(1,len(bounds))
                diff = uB-lB
                I = np.ones((X_design_aux.shape[0],1))
                X_design = np.dot(I,lB) + X_design_aux*np.dot(I,diff)
            except:
                print("Cannot find pyDOE library, please install it to use a Latin hypercube to initialize the model.")

        ## Add random discrete component
        for k in space.get_discrete_dims():
            X_design[:,k] = np.random.choice(space.space_expanded[k]['domain'], data_init)
    
    return X_design

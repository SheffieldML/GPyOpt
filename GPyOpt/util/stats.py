# Copyright (c) 2015, Javier Gonzalez
# Copyright (c) 2015, the GPy Authors (see GPy AUTHORS.txt)
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from ..util.general import samples_multidimensional_uniform


def initial_design(design,space,data_init):
    """
    :param design: the choice of designs
    :param bounds: the boundary of initial points
    :param data_init: the number of initial points
    :Note: discrete dimensions are always added based on uniform samples
    """
    import numpy as np
    bounds = space.get_bounds()
    if design == 'random':
        X_design = samples_multidimensional_uniform(bounds, data_init)
    elif design == 'latin':
        try:
            from pyDOE import lhs
            # Genretate point in unit hypercube
            X_design_aux = lhs(len(bounds),data_init, criterion='center')
            # Normalize to the give box constrains
            lB = np.asarray(bounds)[:,0].reshape(1,len(bounds))
            uB = np.asarray(bounds)[:,1].reshape(1,len(bounds))
            diff = uB-lB
            I = np.ones((X_design_aux.shape[0],1))
            X_design = np.dot(I,lB) + X_design_aux*np.dot(I,diff)
        except:
            print("Cannot find pyDOE library, please install it to use a Latin hypercube to initialize the model.")

    ## Add random dicrete component
    for k in space.get_discrete_dims():
        X_design[:,k] = np.random.choice(space.space_expanded[k]['domain'], data_init)
    return X_design

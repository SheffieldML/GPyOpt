import numpy as np
from scipy.sparse.linalg.eigen.arpack import eigsh as largest_eigsh
from  numpy.linalg import inv
mtb = None

def sample_dpp(L,k=None):
    '''
    Wrapper function for the sample_dpp Matlab code written by Alex Kulesza
    Given a kernel matrix L, returns a sample from a k-DPP.
    The code is hacked in a way that if a set A is provied, samples from a conditional 
    dpp given A are produced
    L:     kernel matrix
    k:     size of the sample from the DPP
    set:   index of the conditional elements. Integer numpy array containing the locations 
            (starting in zero) relative to the rows of L.
       
    '''
    # Matlab link
    global mtb    
    if mtb == None:
        import matlab_wrapper
        mtb = matlab_wrapper.MatlabSession()

    # load values in Matlab and get sample
    mtb.putvalue('L',L)
    if k!=None: 
        k = np.array([[k]])  # matlab only undenstand matrices 
        mtb.put('k',k)
        mtb.eval("dpp_sample = sample_dpp(decompose_kernel(L),k)")
    else:
        mtb.eval("dpp_sample = sample_dpp(decompose_kernel(L))")
        
    #dpp_sample = mtb.getvalue('dpp_sample')
    dpp_sample = mtb.get('dpp_sample')
    return dpp_sample.astype(int)


def sample_conditional_dpp(L,set0,k=None):
    '''
    Wrapper function for the sample_dpp Matlab code written by Alex Kulesza
    Given a kernel matrix L, returns a sample from a k-DPP.
    The code is hacked in a way that if a set A is provied, samples from a conditional 
    dpp given A are produced
    L:     kernel matrix
    set:   index of the conditional elements. Integer numpy array containing the locations 
            (starting in zero) relative to the rows of L.
    k:     size of the sample from the DPP
    '''
    # Calculate the kernel for the marginal
    Id = np.array([1]*L.shape[0])
    Id[set0] = 0
    Id = np.diag(Id)    
    L_compset_full = inv(Id + L)
    L_minor = inv(np.delete(np.delete(L_compset_full,tuple(set0), axis=1),tuple(set0),axis=0))
    L_compset = L_minor - np.diag([1]*L_minor.shape[0])
    
    # Compute the sample
    sample = sample_dpp(L_compset,k)
    if k==2: sample = [sample]
    return np.concatenate((set0,sample) ,axis=0)


def sample_dual_conditional_dpp(L,set0,q,k=None):
    '''
    Wrapper function for the sample_dpp Matlab code written by Alex Kulesza
    Given a kernel matrix L, returns a sample from a dual k-DPP.
    The code is hacked in a way that if a set0 A is provied, samples from a conditional 
    dpp given A are produced
    L:     kernel matrix
    set0:   index of the conditional elements. Integer numpy array containing the locations 
           (starting in zero) relative to the rows of L.
    q:     is the number of used eigenvalues
    k:     size of the sample from the DPP
    '''
    # Calculate the kernel of the marginal
    Id = np.array([1]*L.shape[0])
    Id[set0] = 0
    Id = np.diag(Id)    
    L_compset_full = inv(Id + L)
    L_minor = inv(np.delete(np.delete(L_compset_full,tuple(set0), axis=1),tuple(set0),axis=0))
    L_compset = L_minor - np.diag([1]*L_minor.shape[0]) 
    
    # Take approximated sample
    sample = sample_dual_dpp(L_compset,q,k-1)
    if k==2: sample = [sample]
    return np.concatenate((set0,sample) ,axis=0)



def sample_dual_dpp(L,q,k=None):
    '''
    Wrapper function for the sample_dual_dpp Matlab code written by Alex Kulesza
    Given a kernel matrix L, returns a sample from a k-DPP.
    
    L is the kernel matrix
    q is the number of used eigenvalues
    k is the number of elements in the sample from the DPP
    '''
    # Matlab link
    global mtb    
    if mtb == None:
        import matlab_wrapper
        mtb = matlab_wrapper.MatlabSession()
        
    # Extract the feature matrix from the kernel
    evals, evecs = largest_eigsh(L,q,which='LM')
    B = np.dot(evecs,np.diag(evals))
    
    # load values in Matlab and get sample
    mtb.put('B',B)
    #mtb.putvalue('B',B)
    
    if k!=None: 
        k = np.array([[k]])  # matlab only undernstand matrices 
        mtb.put('k',k)
        mtb.eval("dpp_sample = sample_dual_dpp(B,decompose_kernel(B'*B),k)")
    else:
        mtb.eval("dpp_sample = sample_dual_dpp(B,decompose_kernel(B'*B))")
        
    dpp_sample = mtb.get('dpp_sample')
    return dpp_sample.astype(int)











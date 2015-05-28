import numpy as np
mtb = None


def emin_epmgp(m,K,eta):
    '''
    This is a wrapper function for the matlab function emin_epmgp that calculated the expectation of a Gaussian vector 
    with mean m and covariance K and a costant eta.
    '''
    global mtb    
    if mtb == None:
        import matlab_wrapper
        mtb = matlab_wrapper.MatlabSession()
    mtb.put('K',(K+K.T)*.5+0.01*np.eye(K.shape[0]))  # symetrize and regularize the matrix
    mtb.put('m',m)
    mtb.put('eta',eta)
    mtb.eval("[e_min,Int_y,Probs_y,Int_eta] = emin_epmgp(m,K,eta)")
    e_min = mtb.get('e_min')
    return e_min


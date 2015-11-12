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

    try:
        mtb.put('K',0.5*(K+K.T))  # symetrize to avoid numerical inconsitencies
        mtb.put('m',m)
        mtb.put('eta',eta)
        mtb.eval("[e_min,Int_y,Probs_y,Int_eta] = emin_epmgp(m,K,eta)")
        e_min = mtb.get('e_min')
    except:
        mtb.put('K',0.5*(K+K.T)+0.1*np.diag(np.ones(K.shape[0])))  # regularize in case of errors caused by K being singular
        mtb.put('m',m)
        mtb.put('eta',eta)
        mtb.eval("[e_min,Int_y,Probs_y,Int_eta] = emin_epmgp(m,K,eta)")
        e_min = mtb.get('e_min')
    return e_min


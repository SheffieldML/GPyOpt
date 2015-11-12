from ..dpp_samplers.dpp import sample_dual_conditional_dpp
from ..quadrature.emin_epmgp import emin_epmgp
from ..util.general import samples_multidimensional_uniform, reshape, get_moments, get_quantiles
import numpy as np
import GPyOpt

# def loss_nsahead(x, n_ahead, model, bounds, n_samples_dpp = 5):
#     '''
#     x: 
#     n_ahead: 
#     model: 
#     bounds:
#     current_loss:
#     :param beta: weight of the current loss in the ddp sample
#     '''


#     x = reshape(x,model.X.shape[1]) 
#     n_data = x.shape[0]

#     # --- fixed options
#     num_init_dpp    = 200              # uniform samples
#     n_replicates    = n_samples_dpp    # dpp replicates
#     q               = 50               # truncation, for dual dpp

#     # --- get values
#     losses_samples  = np.zeros((n_data,n_replicates))
#     Y               = model.Y
#     eta             = Y.min()
    
#     X0              = samples_multidimensional_uniform(bounds,num_init_dpp)
#     set1            = [1]
    
    # if n_ahead>1:                  
    #     # --- We need to this separatelly for each data points
    #     for k in range(n_data):
    #         X             = np.vstack((x[k,:],X0))

    #         # --- define kernel matrix for the dpp. Take into account the current loss if available.
    #         # --- diversity
    #         K   = model.kern.K(X)

    #         # --- Quality term
    #         Q   = np.diag(np.exp(-model.predict(X)[0]).flatten())

    #         # --- 
    #         L    = np.dot(np.dot(Q,K),Q)

    #         # --- averages of the dpp samples
    #         for j in range(n_replicates):
    #             # --- take a sample from the dpp (need to re-index to start from zero)
    #             dpp_sample = sample_dual_conditional_dpp(L,set1,q,n_ahead)
    #             dpp_sample = np.ndarray.tolist(np.array(dpp_sample)-1)
    #             print X0[dpp_sample,:]

    #             # evaluate GP at the sample and compute full covariance 
    #             m, K       = model.predict(X0[dpp_sample,:],full_cov=True)
       
    #             # compute the expected loss
    #             losses_samples[k][j]  = emin_epmgp(m,K,eta)

    #     losses = losses_samples.mean(1).reshape(n_data,1)
    
    # elif n_ahead ==1:               
    #     m, s, fmin = get_moments(model, x)
    #     phi, Phi, _ = get_quantiles(0, fmin, m, s)                 # self.acquisition_par should be zero
    #     losses =  fmin + (m-fmin)*Phi - s*phi                      # same as EI excepting the first term fmin

    # return losses


def loss_nsahead(X_star, bounds, expected_loss,n_ahead,L, Min, model):
    '''
    Computes the loss n steps n_ahead for the location x_star
    '''
    if n_ahead ==1:
        X_loss=expected_loss.acquisition_function(X_star)

    else:
        # number of points to evaluate
        n_points = X_star.shape[0]
        X_loss   = np.zeros((n_points,1))
    
        ## --- Current best
        Y               = model.Y
        eta             = Y.min()

        for i in range(n_points):
            # --- Compute future locations
            future_locations = predict_locations(X_star[i,:],bounds,expected_loss,n_ahead, L, Min, model)
        
            # --- Evaluate GP at the sample and compute full covariance
            m, K       = model.predict(future_locations,full_cov=True)

            # --- Compute the expected loss
            X_loss[i,:]       = emin_epmgp(m,K,eta)  # this is a bit slow.
    return X_loss


def predict_locations(x_star, bounds, expected_loss, n_ahead, L, Min, model):
    '''
    Predicts future location evaluations based on a Lipschitz inference criterion
    '''
    ### --- load the recursive method to find the batches
    from GPyOpt.core.acquisition import AcquisitionMP
    from GPyOpt.core.optimization import wrapper_DIRECT, wrapper_lbfgsb
    from GPyOpt.util.general import multigrid

    penalized_loss = AcquisitionMP(expected_loss,transform='softplus')
    penalized_loss.set_model(model)

    ## --- initialize the batch with the putative input
    k = 1
    X_batch = x_star

    ## --- predict the remaining future locations
    while k<n_ahead:
        penalized_loss.update_batches(X_batch,L,Min)
        #new_sample = wrapper_DIRECT(penalized_loss.acquisition_function,bounds)      
        samples = samples_multidimensional_uniform(bounds, 10000)
        pred_samples = penalized_loss.acquisition_function(samples)
        x0 =  samples[np.argmin(pred_samples)]
        new_sample,_ = wrapper_lbfgsb(f=penalized_loss.acquisition_function,grad_f=None,x0 = np.array(x0),bounds=bounds)
        #new_sample = x0
        X_batch = np.vstack((X_batch,new_sample))
        k +=1
    return X_batch

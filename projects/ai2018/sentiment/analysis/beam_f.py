import numpy as np
from beam_f_utils import *

#-----------------  concave optimization -------------------#
def beam_f(N, weights, y,
           bi_concave_iters, conf_opt_iters,
           prob_estimates,
           eps, reg, thresh, last_k,
           out_put=True, seed=0):
    '''
    Args
    -----------------
    N: number of classes
    weights: weights for classes
    y: class labels for data
    bi_concave_iters, conf_opt_iters: inner, outter loop max iterations
    prob_estimates: class probability model (previously trained)
    eps, reg: perturbation constant, regularization
    thresh: convergence threshhold
    last_k: number of objective function values
    out_put: display/suppress output of optimization process
    seed: type of initial confusion matrix
    
    Return
    -----------------
    classifiers: array of classifiers found
    classifier_weights: weights for classifiers found
    perf: macro f-score performance on data set
    '''
    
    # Get initial Gain Matrix
    #--------------   weighted initialization
    if seed == 0:
        G_diag = [weights[n] for n in range(N)]
        G_0 = np.diag(G_diag)
    
    #--------------   balanced initialization
    elif seed == 1:
        G_0 = np.eye(N)
    
    #--------------   random initialization
    else:
        G_0 = 20 * np.random.random((N, N))
    
    # Get the initial confusion matrix corresponding to the classifer
    # for the initial gain matrix
    Conf_t = compute_conf(G_0, prob_estimates, y)
    
    # Find performance of initial confusion matrix
    perf = eval_conf(Conf_t)
    
    #if out_put == True:
    print('initial:', perf, '\n')
    
    # Record all classifiers found across all iterations, along with their weights
    classifiers = np.zeros((bi_concave_iters * conf_opt_iters + 1, N, N))
    classifier_weights = np.zeros(bi_concave_iters * conf_opt_iters + 1)
    
    classifiers[0, : , :] = G_0
    classifier_weights[0] = 1
    
    classifier_index = 1
    
    # Last k objective from outer loop
    outer_obj = []
    
    # Out loop: optimize u for fixed conf mat
    for s in range(bi_concave_iters):
        u_s = compute_u(np.copy(Conf_t), eps)
        
        # Last k objective from inner loop
        inner_obj = []
        
        # Inner loop: optimize conf mat for fixed u
        for t in range(conf_opt_iters):
	        # Compute biconcave objective for the confusion matrix at inner step t
            obj = compute_biconcave_obj(Conf_t, u_s, reg)
            
            #if out_put == True:
            #    print(s, ',', t, ':', perf, ', ', obj)
            #print(t, perf)
            
            # Update last k objectives and terminate inner loop
            # if insufficient improvement in obj function is observed
            if t == conf_opt_iters - 1:
                if len(outer_obj) >= last_k:
                    outer_obj.pop(0)
                    outer_obj.append(obj)
                else:
                    outer_obj.append(obj)
            else:
                if len(inner_obj) >= last_k:
                    inner_obj.pop(0)
                    inner_obj.append(obj)
                    
                    diff = np.absolute(np.array(inner_obj) \
                           - np.array(inner_obj[1:] + [0]))
                    
                    if (diff <= thresh).all():
                        if len(outer_obj) >= 3:
                            outer_obj.pop(0)
                            outer_obj.append(obj)
                        else:
                            outer_obj.append(obj)
                        break
                else:
                    inner_obj.append(obj)
            
            # Compute the gain matrix at inner step t
            G_t = compute_conf_grad(Conf_t, u_s, eps, reg)
            
            # Find new confusion matrix for gain matrix at inner step t
            Conf_new = compute_conf(G_t, prob_estimates ,y)
            
            # Update the confusion matrix using line search
            max_perf = -1
            step_size = 0
            
            for i in range(100):
                l = i * 0.01
                Conf_line = l * Conf_new + (1 - l) * Conf_t
                perf_line = compute_biconcave_obj(Conf_line, u_s, reg)
                
                if perf_line > max_perf:
                    max_perf = perf_line
                    Conf_t[:, :] = Conf_line
                    step_size = l
            
            # Record the gain matrix and the corresponding weight
            classifiers[classifier_index, :, :] = G_t
            classifier_weights[:classifier_index] = classifier_weights[:classifier_index] * (1 - step_size)
            classifier_weights[classifier_index] = step_size
            classifier_index = classifier_index + 1
            
            # Find performance of confusion matrix accepted at inner step t
            perf = eval_conf(Conf_t)
        
        # Terminate inner loop if insufficient improvement in
        # obj function is observed
        diff = np.absolute(np.array(outer_obj) \
               - np.array(outer_obj[1:] + [0]))
        
        print(perf, diff)
        if (diff <= thresh).all():
            break
    
    # Truncate the classifer and classifier weights arrays
    classifiers = classifiers[:classifier_index, :, :]
    classifier_weights = classifier_weights[:classifier_index]
    
    #if out_put == True:
    print('final: ', str(perf), '\n')
    
    return classifiers, classifier_weights, perf


def seed_beam_f(N, weights, y,
                bi_concave_iters, conf_opt_iters,
                prob_estimates,
                eps, reg, thresh, last_k,
                out_put=None, restarts=5):
    
    '''
    Args
    -----------------
    N: number of classes
    weights: weights for classes
    y: class labels for data
    bi_concave_iters, conf_opt_iters: inner, outter loop max iterations
    prob_estimates: class probability model (previously trained)
    eps, reg: perturbation constant, regularization
    thresh: convergence threshhold
    last_k: number of objective function values
    out_put: display/suppress output of optimization process
    restarts: number of restarts to use for beam_f
    
    Return
    -----------------
    best_classifiers: best classifiers found using restarts
    best_weights: weights for best classifiers found using restarts
    '''
    
    best_classifiers = np.zeros((bi_concave_iters * conf_opt_iters + 1, N, N))
    best_weights = np.zeros(bi_concave_iters * conf_opt_iters + 1)
    best_perf = -1
    
    # Call beam_f for multiple restarts with different types
    # of initial gain matrices
    for i in range(restarts):
        classifiers, classifier_weights, perf = beam_f(N, weights, y,
                                                       bi_concave_iters, conf_opt_iters,
                                                       prob_estimates,
                                                       eps, reg, thresh, last_k,
                                                       out_put=out_put, 
                                                       #seed=i)
                                                       #seed=1)
                                                       #seed=0)
                                                       seed=2)

        print(perf)
        # if perf > best_perf:
        #     best_classifiers[:, :, :] = classifiers
        #     best_weights[:] = classifier_weights
        #     best_perf = perf
    
    return best_classifiers, best_weights
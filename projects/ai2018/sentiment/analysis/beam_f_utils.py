import numpy as np

#------------  compute objective function ---------------#
def compute_biconcave_obj(C, u, reg):
    '''
    Computes the biconcave objective function
	 
    Args
    -----------------
    C: confusion matrix
    u: value of u
    reg: regularization parameter
    
    Return
    -----------------
    obj: biconcave objective evaluate at u and confusion matrix
    '''

    # Number of classes
    N = C.shape[0]

    P_Conf = np.array([np.sum(C[n, :]) + np.sum(C[:, n]) for n in range(N)])

    obj = 2 * (u.dot(np.sqrt(np.diag(C)))) - (u**2).dot(P_Conf) - reg * np.linalg.norm(P_Conf)
    
    return obj

#------------  compute performance for conf matrix  --------------#
def compute_u(C, eps):
    '''
    Optimizes u for a fixed confusion matrix C
    
    Args
    -----------------
    Conf: confusion matrix
    eps: perturbation parameter
    
    Return
    -----------------
    u: optimal u for confusion matrix C
    '''
    
    # Adding a small perturbation helps avoid loc optima
    C += eps
    # Number of classes
    N = C.shape[0]
    
    u = np.zeros(N)
    
    # Update u using closed form expression
    for n in range(N):
        u[n] = np.sqrt(C[n, n]) / (np.sum(C[n, :]) + np.sum(C[:, n]))
    
    return u

#------------  evaluate performance of classifier  --------------#
def eval_classifier(prob_estimates, G, X, y):
    '''
    Evaluate performance of classifier in terms of f-score
    
    Args
    -----------------
    prob_estimates: probability estimates for X
    G: gain matrix
    X: attributes
    y: labels
    
    Return
    -----------------
    perf: performance of classifier
    Conf: confusion matrix of classifier
    '''
    
    Conf = compute_conf(G, prob_estimates, y)
    perf = eval_conf(Conf)
    
    return perf, Conf

#------------  compute performance for conf matrix  --------------#
def eval_conf(C): # updated to use f-score
    '''
    Evaluate performance of confusion matrix in terms of f-score
    
    Args
    -----------------
    C: confusion matrix
    
    Return
    -----------------
    macro f-score of confusion matrix
    '''
    # Number of classes
    N = C.shape[0]
    perf = 0
    cl = 0
    
    for n in range(N):
        if (np.sum(C[n, :]) + np.sum(C[:, n])) > 0:
            perf = perf + (C[n, n] * 2.) / (np.sum(C[n, :]) + np.sum(C[:, n]))
            cl += 1
        #else:
            #print 'invalid confusion matrix at class', n
            #sys.exit('invalid confusion matrix')
    if cl == 0:
        return 0
    else:
        return perf / (1. * cl)
 

#------------  gradient of performance measure  -------------#
def compute_conf_grad(C, u, eps, reg): # updated to use f-score
    '''
    Finds gradient of f-score at confusion matrix C
    
    Args
    -----------------
    C: confusion matrix
    u: value of u
    eps: perturbation parameter
    reg: regularization parameter
    
    Return
    -----------------
    gradient of confusion matrix
    '''
    # Number of classes
    N = C.shape[0]
    C += eps
    
    grad = np.zeros(C.shape)
    
    for n in range(N):
        P = np.zeros(C.shape)
        P[n, :] = 1
        P[:, n] = 1
        P[n, n] = 2
        grad = grad - u[n] * u[n] * P
        grad[n, n] = grad[n, n] + u[n] / np.sqrt(2. * C[n, n])
    
    return grad - reg * 2 * C

#-----------------  predict labels -------------------#
def predict_labels(G, eta):
    '''
    Outputs prediction of classifier with gain matrix G
    
    Args
    -----------------
    G: gain matrix
    eta: probability estimates
    
    Return
    -----------------
    labels: labels
    '''
    
    M = eta.shape[1]
    
    # optimal class labels
    labels = np.zeros(M)
    
    for m in range(M): # for each data point
        eta_m = eta[:, m] # get eta for the m^{th} point
        t = G.dot(eta_m) # get a row vector with (g_{y})'*\eta_{x}
        indx = np.argmax(t) # weighted argmax
        labels[m] = indx # label for m-th data point
    
    return labels

#-----------------  compute confusion matrix -------------------#
def compute_conf(G, eta, true_labels):
    '''
    Given a gain matrix G computes its confusion matrix C
    
    Args
    -----------------
    G: gain matrix
    eta: probability estimates
    true_labels: true labels
    
    Return
    -----------------
    C: confusion matrix
    '''
    
    # Number of classes
    N = G.shape[0]
    # Number of instances
    M = len(true_labels)
    
    # Initialize nxn Confision matrix
    C = np.zeros(G.shape)
    
    # Get prediction for the given classifier (gain matrix G)
    pred_labels = predict_labels(G, eta)
    pred_labels = pred_labels.reshape((pred_labels.shape[0], 1))
    
    # Update the entries of confusion matrix
    for i in range(N):
        for j in range(N):
            comp_labels = pred_labels[true_labels == i]
            C[i, j] = len(comp_labels[comp_labels == j])
    
    C = C / (M * 1.)
    return C

def compute_rand_conf(classifiers, classifier_weights, eta, true_labels):
    '''
    Given array of classifiers and corresponding weights computes a radomnized classifier
    
    Args
    -----------------
    classifiers: array of classifiers
    classifier_weights: array of classifier weights
    eta: probability estimates
    true_labels: true labels
    
    Return
    -----------------
    C: confusion matrix
    '''
    # Number of classifiers, classes
    M = classifiers.shape[0]
    N = classifiers.shape[1]
    
    # Expected conf matrix
    C = np.zeros((N, N))
    for m in range(M):
        conf = compute_conf(classifiers[m, :, :].reshape(N, N), eta, true_labels)
        C =  C + classifier_weights[m] * conf
    
    return C

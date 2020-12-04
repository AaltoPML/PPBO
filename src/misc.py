import numpy as np
import scipy
import sklearn #covariance.shrunk_covariance
import sklearn.covariance


#''' Scaling '''
#from sklearn.preprocessing import MinMaxScaler
#def scale(X_, x_min=0, x_max=1): #old x_min=-1, x_max=1
#    nom = (X_-X_.min(axis=0))*(x_max-x_min)
#    denom = X_.max(axis=0) - X_.min(axis=0)
#    denom[denom==0] = 1
#    return x_min + nom/denom
#def unscale(X_, X_nonscaled,x_min=0, x_max=1):
#    Xmin = X_nonscaled.min(axis=0)
#    Xmax = X_nonscaled.max(axis=0)
#    nom = (X_-x_min)*(Xmax-Xmin)
#    denom = (x_max-x_min)
#    return Xmin + nom/denom




''' Find bounds for alpha '''
def alpha_bounds(xi,lower,upper):
    xi = np.array(xi)
    lower = np.array(lower)
    upper = np.array(upper)
    #alpha_lower = np.max([np.max(lower_bounds[xi>0]/xi[xi>0]),np.max(upper_bounds[xi<0]/xi[xi<0])])
    #alpha_upper = np.min([np.min(lower_bounds[xi<0]/xi[xi<0]),np.min(upper_bounds[xi>0]/xi[xi>0])])
    
    try:
        l1 = np.max(lower[xi>0]/xi[xi>0])
    except:
        l1 = -np.inf
    try:
        l2 = np.max(upper[xi<0]/xi[xi<0])
    except:
        l2 = -np.inf
    alpha_lower = np.max([l1,l2])
    
    try:
        u1 = np.min(lower[xi<0]/xi[xi<0])
    except:
        u1 = np.inf
    try:
        u2 = np.min(upper[xi>0]/xi[xi>0])
    except:
        u2 = np.inf
    alpha_upper = np.min([u1,u2])
    
    if alpha_lower > alpha_upper:
        print("Error: alpha_min > alpha_max!")
    if alpha_lower == -np.inf:
        print("Error: alpha_min is -infinity!")
    if alpha_upper == np.inf:
        print("Error: alpha_max is infinity!")
            
    return alpha_lower, alpha_upper







''' Matrix algebra auxiliary functions '''

def regularize_covariance(X,reg_level=1e-4,pos_diag=True,jitter=1e-7):
    #Force diagonal to positive plus jitter term
    if pos_diag:
        d = np.diag(X).copy()
        d[d<0]=jitter
        np.fill_diagonal(X,d)
    #Regularize by using SVD
    try:
        u, s, vh = np.linalg.svd(X,full_matrices=False)
        X = np.matmul(np.matmul(u, np.diag(s)), vh)
    except:
        pass
    #Regularize by using shrinkage
    try:
        X=sklearn.covariance.shrunk_covariance(X,shrinkage=reg_level)
    except:
        pass
    return(X)
    

def inverse(matrix):
    n = matrix.shape[0]
    return scipy.linalg.solve(matrix, np.identity(n), sym_pos = False, overwrite_b = True)


def pd_inverse(matrix):
    ''' inverse for positive definite matrices '''
    ''' In Scipy, the linalg.solve() function has a parameter sym_pos that assumes the matrix is p.d. '''
    n = matrix.shape[0]
    return scipy.linalg.solve(matrix, np.identity(n), sym_pos = True, overwrite_b = True)


def det(matrix,regularization_level=0):
    ''' Compute determinant by using SVD and logratihm'''
    regularization_intensity = np.max(np.diag(matrix))*regularization_level
    matrix = matrix + np.diag(np.ones((len(matrix))))*regularization_intensity
    U, s, V_star = np.linalg.svd(matrix, full_matrices=False)
    sign_U,det_U = np.linalg.slogdet(U)
    sign_S,det_S = np.linalg.slogdet(np.diag(s))
    sign_V_star,det_V_star = np.linalg.slogdet(V_star)
    det_ = (sign_U*np.exp(det_U))*(sign_S*np.exp(det_S))*(sign_V_star*np.exp(det_V_star))
    return det_

def pseudo_det(matrix):
    eig_values = np.linalg.eig(matrix)[0]
    eig_values = eig_values[np.real(eig_values) > 1e-12] #take positive real-part eigenvalues
    eig_values = eig_values[:min(len(eig_values),300)] #take at most 300 first eigen values
    return np.abs(np.product(eig_values))

def is_positive_definite(M):
    try:
        np.linalg.cholesky(M)
        return True 
    except np.linalg.LinAlgError:
        print('Function is_positive_definite: Matrix is not positive definite!')
        return False
    
def std_normal_pdf(x):
    return (1/(np.sqrt(2*np.pi)))*np.exp(-0.5*np.power(x,2))

#def std_normal_pdf_der(x):
#    return (-x/(np.sqrt(2*np.pi)))*np.exp(-0.5*np.power(x,2))

def var2_normal_pdf(x):
    return (1/(np.sqrt(4*np.pi)))*np.exp(-0.25*np.power(x,2))

#def var2_normal_pdf_der(x):
#    return (-x/(4*np.sqrt(np.pi)))*np.exp(-0.25*np.power(x,2))

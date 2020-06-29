import numpy as np
import time
#import multiprocessing as mp
from GPyOpt.methods import BayesianOptimization



''' Wrapper function to compute next query given GP_model and the type of acquisition function '''
def next_query(PPBO_settings,GP_model,unscale=True):
    start = time.time()
    
    ''' How next dims for xi are determined? EI and EXR'''
    if PPBO_settings.xi_acquisition_function=='EI' or PPBO_settings.xi_acquisition_function=='EXR':      
        #Cyclically w.r.t previous dims? 
        xi_dims = list((np.array(PPBO_settings.xi_dims_prev_iter) + 1)%PPBO_settings.D)
        PPBO_settings.xi_dims_prev_iter = xi_dims
        #Ranomly: random coordinates and random number of non-zero coordinates?
        #xi_dims = list(set(np.random.choice(PPBO_settings.D,PPBO_settings.D-1,replace=True)))
        #Random: random D/2 coordinates
        #xi_dims = list(np.random.choice(PPBO_settings.D,int(PPBO_settings.D/2)))
    
    if PPBO_settings.xi_acquisition_function=='EI': #expceted improvement by projective preferential query
        xi_next, x_next = maximize_EI(xi_dims,GP_model,PPBO_settings)
    elif PPBO_settings.xi_acquisition_function=='EXR': #explore, i.e. varmax
        xi_next, x_next = maximize_varmax(xi_dims,GP_model,PPBO_settings)
    elif PPBO_settings.xi_acquisition_function=="RAND": #random
        xi_next = random_next_xi(PPBO_settings)
        x_next = next_x_given_xi(xi_next,GP_model,PPBO_settings)
    elif PPBO_settings.xi_acquisition_function=="PCD": #preferential coordinate descent
        xi_next = PCD_next_xi(PPBO_settings)
        x_next = next_x_given_xi(xi_next,GP_model,PPBO_settings)
    elif PPBO_settings.xi_acquisition_function=="EXT": #exploit
        xi_next = EXT_next_xi(PPBO_settings,GP_model)
        x_next = next_x_given_xi(xi_next,GP_model,PPBO_settings)
    else:
        print('Invalid acquisition function name!')
        return 0
    if GP_model.verbose: print("Evaluation of the acquisition function took " + str(time.time()-start) + " seconds.")
    
    ''' Normalize xi before unscaling! '''
    xi_next =  np.abs(xi_next)/np.max(np.abs(xi_next))
    if unscale:
        xi_next = GP_model.FP.unscale(xi_next,retain_0_values=True)
        x_next = GP_model.FP.unscale(x_next,retain_0_values=True)       
        if GP_model.verbose: print("Next query: (xi,x) = " + str((xi_next,x_next)))
        return (xi_next,x_next)
    else:
        return (xi_next,x_next)





''' Expected Improvement by projective preferential query (EI) '''
def EI(xi,x,GP_model,mc_samples):
    m = 100 
    xi_grid = GP_model.FP.xi_grid(xi=xi,x=x,alpha_grid_distribution='evenly',alpha_star=None,m=m,is_scaled=True)
    f_post_mean,f_post_covar = GP_model.mu_Sigma_pred(xi_grid)
    mu_star = GP_model.mu_star_
    z = [0]*mc_samples
    for i in range(0,mc_samples):
        f_max = np.max(np.random.multivariate_normal(f_post_mean,f_post_covar)) #predict/sample GP
        z[i] = np.max([f_max-mu_star,0])   
    return -np.mean(z)

def EI_to_maximize(xi_plus_x,xi_dims,x_dims,GP_model,mc_samples):
    xi_plus_x = xi_plus_x[0] #THIS IS NEEDED ONLY IF BO IS THE OPTIMIZER. OTHERWISE COMMENT THIS OUT!
    x = np.zeros(GP_model.D)
    xi = x.copy()    
    xi[xi_dims] = xi_plus_x[xi_dims]
    x[x_dims] = xi_plus_x[x_dims]
    return EI(xi,x,GP_model,mc_samples)
    

def maximize_EI(xi_dims,GP_model,PPBO_settings):
    x_dims = [i for i in range(GP_model.D) if not i in xi_dims]
    
    #BayesianOptimization
    bounds = [{'name': 'var_'+str(d), 'type': 'continuous', 'domain': (0,1)} for d in range(1,GP_model.D+1)]
    BO = BayesianOptimization(lambda xi_plus_x: EI_to_maximize(xi_plus_x, xi_dims,x_dims,GP_model,PPBO_settings.mc_samples), 
                              domain=bounds,
                              optimize_restarts = 1,
                              normalize_Y=True)
    BO.run_optimization(max_iter = PPBO_settings.BO_maxiter)
    res = BO.x_opt     
    x = np.zeros(GP_model.D)
    xi = np.zeros(GP_model.D)
    xi[xi_dims] = res[xi_dims]
    x[x_dims] = res[x_dims]
    xi = perturbate_zerocoordinates(xi,xi_dims)
    x = perturbate_zerocoordinates(x,x_dims)
    return xi, x

''' --------------------------------------'''




''' Variance maximization by projective query (varmax) '''
def varmax(xi,x,GP_model,mc_samples):
    m = 100
    xi_grid = GP_model.FP.xi_grid(xi=xi,x=x,alpha_grid_distribution='evenly',alpha_star=None,m=m,is_scaled=True)
    f_post_mean,f_post_covar = GP_model.mu_Sigma_pred(xi_grid)
    z = [0]*mc_samples
    for i in range(0,mc_samples):
        f_max = np.max(np.random.multivariate_normal(f_post_mean,f_post_covar)) #predict/sample GP
        z[i] = f_max
    return -float(np.mean(np.power(z-np.mean(z),2)))

def varmax_to_maximize(xi_plus_x,xi_dims,x_dims,GP_model,mc_samples):
    xi_plus_x = xi_plus_x[0] #THIS IS NEEDED ONLY IF BO IS THE OPTIMIZER. OTHERWISE COMMENT THIS OUT!
    x = np.zeros(GP_model.D)
    xi = x.copy()    
    xi[xi_dims] = xi_plus_x[xi_dims]
    x[x_dims] = xi_plus_x[x_dims]
    return varmax(xi,x,GP_model,mc_samples)
    

def maximize_varmax(xi_dims,GP_model,PPBO_settings):
    x_dims = [i for i in range(GP_model.D) if not i in xi_dims]
    
    #BayesianOptimization
    bounds = [{'name': 'var_'+str(d), 'type': 'continuous', 'domain': (0,1)} for d in range(1,GP_model.D+1)]
    BO = BayesianOptimization(lambda xi_plus_x: varmax_to_maximize(xi_plus_x, xi_dims,x_dims,GP_model,PPBO_settings.mc_samples), 
                              domain=bounds,
                              optimize_restarts = 1,
                              normalize_Y=True)
    BO.run_optimization(max_iter = PPBO_settings.BO_maxiter)
    res = BO.x_opt     
    x = np.zeros(GP_model.D)
    xi = np.zeros(GP_model.D)
    xi[xi_dims] = res[xi_dims]
    x[x_dims] = res[x_dims]
    xi = perturbate_zerocoordinates(xi,xi_dims)
    x = perturbate_zerocoordinates(x,x_dims)
    return xi, x

''' --------------------------------------'''




def random_next_xi(PPBO_settings):
    nonzero_coords = list(set(np.random.choice(PPBO_settings.D,PPBO_settings.D-1,replace=True))) 
    #i.e. choose D-1 random numbers from a set {0,...,D-1} WITH REPLACEMENT (SO NUMBER OF NON-ZERO COORDINATES MAY VARY)
    xi_next = np.zeros(PPBO_settings.D)
    xi_next[nonzero_coords] = np.random.uniform(0, 1, (1, len(nonzero_coords)))[0]
    return xi_next

def PCD_next_xi(PPBO_settings):
    I = np.eye(PPBO_settings.D)
    d = int(PPBO_settings.dim_query_prev_iter + 1)
    if d > PPBO_settings.D:
        d = 1
    PPBO_settings.dim_query_prev_iter = d
    return I[:,d-1]

def EXT_next_xi(PPBO_settings,GP_model):
    x_star = GP_model.x_star_.copy()
    x_star[x_star==0] = 1e-7 #If some coordinate is zero, add epsilon to selections of type where(x_star=0) works.
    d = int(PPBO_settings.dim_query_prev_iter + 1)
    if d > PPBO_settings.D:
        d = 1
    PPBO_settings.dim_query_prev_iter = d
    xi_next = x_star
    xi_next[d-1] = 0
    return xi_next




def next_x_given_xi(xi,GP_model,PPBO_settings):
    '''' Function finds optimal next x by maximizing aquisition function'''  
    nonzero_coords = list(np.where(xi==0)[0])
    x_next = np.zeros(PPBO_settings.D)
    if PPBO_settings.x_acquisition_function == "exploit":
        x_star = GP_model.x_star_.copy()
        x_next[nonzero_coords] = x_star[nonzero_coords]
    elif PPBO_settings.x_acquisition_function == "random":
        x_next[nonzero_coords] = np.random.uniform(0, 1, (1, len(nonzero_coords)))[0]
    else:
        print("Invalid acquisition function selected!")
        return None
    
    x_next = perturbate_zerocoordinates(x_next,nonzero_coords)                     
    
    return x_next


def perturbate_zerocoordinates(x,nonzero_coords):
    #If exactly zero coordinate values
    x_ = x[nonzero_coords].copy()
    x_[x_==0] = 1e-7 #amount of perturbation
    x[nonzero_coords] = x_
    return x


    
    

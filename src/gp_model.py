from feedback_processing import FeedbackProcessing

import numpy as np
import scipy
import scipy.stats

from scipy.special import ndtr as std_normal_cdf #scipy.special.ndtr fast numerical integration for standard normal cdf
from scipy.integrate import quad as integrate  #numerical integration
from GPyOpt.methods import BayesianOptimization #Use as global optimizer e.g. fro evidence maximization

from itertools import chain #To unlist lists of lists
import time

from kernels import SE_kernel, RQ_kernel, camphor_copper_kernel
from misc import inverse, is_positive_definite, pd_inverse, std_normal_pdf, var2_normal_pdf, pseudo_det, det, regularize_covariance


class GPModel:
    """
    Class for GP utility function model.
    Some methods are not inherited but composited from Feedback_Processing class
    """
    
    COVARIANCE_SHRINKAGE = 7e-6 #An amount of shrinkage applied to Sigma. DEFAULT: 1e-6 
    #Low value is better but is numerically more unstable
    #Note! Higher value increases difference to random Fourier features approximation of f
    @staticmethod
    def set_COVARIANCE_SHRINKAGE(shrinkage=7e-6):
        COVARIANCE_SHRINKAGE=shrinkage

    def __init__(self, PPBO_settings):
        """
        Initializes the GP_Model object
        """
        self.verbose = PPBO_settings.verbose
        self.FP = None
        
        self.D = PPBO_settings.D  #Problem dimension 
        self.original_bounds = PPBO_settings.original_bounds #Boundaries of each variables as a sequence of tuplets
        self.bounds = ((0,1),)*self.D
        
        self.X = None   #Design Matrix
        self.N = None   #Number of observations including pseudo-observations
        self.m = PPBO_settings.n_pseudoobservations #How many pseudo-observations per one observation?
        self.obs_indices = None  #Locations of true observations
        self.pseudobs_indices = None  #Locations of pseudo-observations
        self.latest_obs_indices = None  #Location of latest true observation given all observations from 0 to N
        self.alpha_grid_distribution = PPBO_settings.alpha_grid_distribution   #equispaced, Cauchy or TGN
        self.TGN_speed = PPBO_settings.TGN_speed
        self.n_gausshermite_sample_points = PPBO_settings.n_gausshermite_sample_points
        self.xi_acquisition_function = PPBO_settings.xi_acquisition_function
  
        self.kernel = eval(PPBO_settings.kernel) #Kernel type
        self.theta_initial = PPBO_settings.theta_initial
        self.theta = None #Most optimized theta
        self.Sigma = None
        self.Sigma_inv = None
        self.Lambda_MAP = None
        self.posterior_covariance = None
         
        self.fMAP = None
        self.max_iter_fMAP_estimation = PPBO_settings.max_iter_fMAP_estimation
        self.fMAP_finding_trials = 1
        self.fMAP_optimizer = PPBO_settings.fMAP_optimizer
        self.fMAP_random_initial_vector = True
        self.mustar_finding_trials = PPBO_settings.mustar_finding_trials
        self.mustar_previous_iteration = 0
        self.mustar = None #Global maximum of (predictive mean) utility function
        self.xstar = None #Global maximizer of (predictive mean) utility function
        self.xstars_local = None #Local maximizers of utility function observed during the optimization
        
        self.initialization_running = True #Less intensive computations during initialization (if skip_... = True)
        self.last_iteration = False #Super intensive computations at the last iteration
        self.skip_computations_during_initialization = PPBO_settings.skip_computations_during_initialization
        
 
    ''' --- Wrapper functions --- '''
    def update_feedback_processing_object(self,X_obs):
        if self.FP is None:
            self.FP = FeedbackProcessing(self.D, self.m,self.original_bounds,self.alpha_grid_distribution,self.TGN_speed)
            self.FP.initialize_data(X_obs)
        else:
            self.FP.update_data(X_obs)
    
    def update_data(self):
        self.X = self.FP.X
        self.N = self.FP.N
        self.obs_indices = self.FP.obs_indices  
        self.pseudobs_indices = self.FP.pseudobs_indices  
        self.latest_obs_indices = self.FP.latest_obs_indices
    
    def update_model(self,optimize_theta=False):
        if self.theta is None:
            self.set_theta() 
        self.update_Sigma(self.theta)
        self.update_Sigma_inv(self.theta)
        if self.initialization_running and self.skip_computations_during_initialization:
            self.FP.alpha_grid_distribution = 'equispaced' #Always use 'equispaced' pseudo-observations in initialization
            self.update_fMAP(random_initial_vector=False,fmap_finding_trials=1)
        elif self.last_iteration:
            self.update_fMAP(random_initial_vector=True,fmap_finding_trials=5)
        else:
            self.update_fMAP()
        if optimize_theta:
            self.optimize_theta()
            self.update_fMAP()
            self.update_Sigma(self.theta)
            self.update_Sigma_inv(self.theta)
        if self.verbose: print("Current theta is: " + str(self.theta) + ' (Acq. = ' +str(self.xi_acquisition_function)+')')
        if self.verbose: print("Updating Lambda_MAP...")
        start = time.time()
        self.Lambda_MAP = self.create_Lambda(self.fMAP,self.theta[0])
        if self.verbose: print("... this took " + str(time.time()-start) + " seconds.")
        if self.verbose: print("Updating posterior covariance...")
        start = time.time()
        try:
            self.posterior_covariance_inv = self.Sigma_inv - self.Lambda_MAP #self.Sigma_inv + self.Lambda_MAP
            self.posterior_covariance = pd_inverse(self.posterior_covariance_inv)
        except:
            print('---!!!--- Posterior covariance matrix is not PSD ---!!!---')
            pass
        if self.verbose: print("... this took " + str(time.time()-start) + " seconds.")
        if self.verbose: print("Computing mu_star and x_star ...")
        start = time.time()
        if self.initialization_running and self.skip_computations_during_initialization:
            self.xstar, self.mustar, self.xstars_local = self.mu_star(mustar_finding_trials=1)
        elif self.last_iteration:
            self.xstar, self.mustar, self.xstars_local = self.mu_star(mustar_finding_trials=5)    
        else:
            self.xstar, self.mustar, self.xstars_local  = self.mu_star()
        if self.verbose: print("... this took " + str(time.time()-start) + " seconds.")
    
    ''' Auxiliary function '''
    def turn_initialization_off(self):
        self.initialization_running = False
        self.FP.alpha_grid_distribution = self.alpha_grid_distribution
        
    def set_last_iteration(self):
        self.last_iteration = True
    
    def is_pseudobs(self,i):
        return self.FP.is_pseudobs(i)
    
    ''' --- Covariance matrix --- '''
    
    @staticmethod
    def create_Gramian(X1,X2,kernel,*args): 
        Sigma = kernel(X1,X2, *args)
        ''' REGULARIZATION OF THE COVARIANCE MATRIX'''
        Sigma = regularize_covariance(Sigma,GPModel.COVARIANCE_SHRINKAGE)
        return Sigma

    @staticmethod
    def create_Gramian_nonsquare(X1,X2,kernel,*args): 
        Sigma = kernel(X1,X2, *args)
        return Sigma

    def update_Sigma(self,theta):
        self.Sigma = self.create_Gramian(self.X,self.X,self.kernel,theta)
        #print("The condition number of the covariance matrix: " + str(np.linalg.cond(self.Sigma,p=np.inf))) #condition number of A = ||A||*||A^-1||, where the norm ||.|| is the maximum absolute row sum  (since p =np.inf)
        
    def update_Sigma_inv(self,theta): 
        self.Sigma_inv = pd_inverse(self.Sigma)
    
    def set_theta(self):
        self.theta = self.theta_initial
        if self.theta[1] is None:
            self.theta[1] = 1
        if self.theta[2] is None:
            self.theta[2] = 0.01
        if self.theta[0] is None:
            self.theta[0] = 0.01 
    
    ''' --- Functional T --- '''
    
    ''' Auxiliary functions for computing transormations/vectroizations of Phi '''    
    def sum_Phi(self,i,order_of_derivative,f,sigma,sample_points=None, weights=None):
        '''
        Auxiliary function that computes summation of Phi-function values
        i = index of x observation, i.e. some element of list 'obs_indices'
        order_of_derivative = how many this the c.d.f of the standard normal (Phi) is differentiated?
        f = vector of f-values
        '''
        m = self.m
        #Delta_i_j = (f[i+j+1]-f[i])/(sigma_)
        Delta = f[i+1:i+m+1]
        Delta = (Delta - f[i])/sigma
        sum_=0
        if order_of_derivative==0:
            for j in range(0,m):
                #sum_ = sum_ + integrate(lambda x: std_normal_cdf(Delta[j]+x)*std_normal_pdf(x), -np.inf, np.inf)[0]
                #Do integration by using Gaussian-Hermite quadrature: 
                sum_ = sum_ + (1/np.sqrt(np.pi))*np.dot(weights,std_normal_cdf(Delta[j]-np.sqrt(2)*sample_points)) #Do to change of variables to get integteragl into the form int exp(x^2)*....dx. This is why np.sqrt(2)
            return(sum_)
        if order_of_derivative==1:
            for j in range(0,m):
                sum_ = sum_ + float(var2_normal_pdf(Delta[j]))
            return(sum_)
        if order_of_derivative==2:
            for j in range(0,m):
                sum_ = sum_ - 0.5*Delta[j]*float(var2_normal_pdf(Delta[j]))
            return(sum_)
        else:
            print("The derivatives of an order higher than 2 are not needed!")
            return None
        
    def sum_Phi_vec(self,order_of_derivative,f,sigma,over_all_indices=False):
        '''
        Auxiliary function that create a vector of sum_Phi with elements as 
        i = obs_indices[0],...,obs_indices[len(obs_indices)],   if not over_all_indices
        i = 1,...,N                                      if over_all_indices
        '''
        sample_points, weights = np.polynomial.hermite.hermgauss(self.n_gausshermite_sample_points) #for cross-correlation integral 
        if not over_all_indices:
            sum_Phi_vec_ = [self.sum_Phi(i,order_of_derivative,f,sigma,sample_points, weights) for i in self.obs_indices]
            return np.array(sum_Phi_vec_)
        else: 
            sum_Phi_vec_ = list(chain.from_iterable([[self.sum_Phi(i,order_of_derivative,f,sigma,sample_points, weights)]*(self.m+1) for i in self.obs_indices])) #for z indices just replicate previous x value
            return np.array(sum_Phi_vec_)

    ''' Derivatives of the functional T of the order: 0,1,2 '''
    def T(self,f,theta,Sigma_inv_=None):
        if Sigma_inv_ is None: 
            Sigma_inv_=self.Sigma_inv
        sumPhi = self.sum_Phi_vec(0,f,theta[0])
        T = -0.5*f.T.dot(Sigma_inv_).dot(f) - np.sum(sumPhi)/self.m
        return T
    
    def T_grad(self,f,theta,Sigma_inv_=None):
        if Sigma_inv_ is None: 
            Sigma_inv_=self.Sigma_inv
        N = self.N
        m = self.m
        latest_obs_indices = self.latest_obs_indices
        beta = np.zeros((N,1)).reshape(N,)
        Phi_der_vec_ = np.array([float(var2_normal_pdf((f[j]-f[latest_obs_indices[j]])/theta[0])) for j in self.pseudobs_indices])
        sum_Phi_der_vec_ = self.sum_Phi_vec(1,f,theta[0])
        beta[self.obs_indices] = sum_Phi_der_vec_/(theta[0]*m)
        beta[self.pseudobs_indices] = -Phi_der_vec_/(theta[0]*m)
        T_grad = -Sigma_inv_.dot(f).reshape(N,) + beta.reshape(N,)
        return T_grad
    
    def T_hessian(self,f,theta,Sigma_inv_=None):
        if Sigma_inv_ is None: 
            Sigma_inv_=self.Sigma_inv
        Lambda = self.create_Lambda(f,theta[0])
        T_hessian = -Sigma_inv_ + Lambda
        return T_hessian

    def create_Lambda(self,f,sigma): 
        N = self.N
        m = self.m
        sample_points, weights = np.polynomial.hermite.hermgauss(self.n_gausshermite_sample_points)
        ''' Diagonal matrix '''
        Lambda_diagonal = [0]*N
        constant = 1/(m*(sigma**2))
        for i in range(N):
            if not self.is_pseudobs(i):
                Lambda_diagonal[i] = -constant*self.sum_Phi(i,2,f,sigma)     #Note that sum_Phi used so minus sign needed!       
            else:
                latest_x_index = np.max([ind for ind in self.obs_indices if ind < i])
                Delta_i = (f[i]-f[latest_x_index])/sigma
                Lambda_diagonal[i] = 0.5*constant*Delta_i*var2_normal_pdf(Delta_i)        
        Lambda_diagonal  = np.diag(Lambda_diagonal)
        ''' Upper triangular off-diagonal matrix'''
        Lambda_uppertri = np.zeros((N,N))
        for row_ind in range(N):
            if not self.is_pseudobs(row_ind):
                for col_ind in range(row_ind+1,row_ind+m+1):   
                    if self.is_pseudobs(col_ind):
                        Delta = (f[col_ind]-f[row_ind])/sigma
                        Lambda_uppertri[row_ind,col_ind] = -0.5*constant*Delta*var2_normal_pdf(Delta)
        ''' Final Lambda is sum of diagonal + upper_tringular  + lower_tringular '''
        Lambda = Lambda_diagonal + Lambda_uppertri + Lambda_uppertri.T
        return Lambda

    ''' --- Evidence --- '''
    
    def evidence(self,theta,f_initial):
        def prior(theta):
            ''' Hyperparameter priors. This stabilizes the optimization '''
            #https://keisan.casio.com/exec/system/1180573226
            priortheta0 = scipy.stats.beta.pdf(theta[0], 0.0005, 35) #loc around 0
            priortheta1 = scipy.stats.beta.pdf(theta[1], 15, 35) #loc around 0.3
            priortheta2 = scipy.stats.beta.pdf(theta[2], 3, 20) #loc around 0.1
            return np.log(priortheta0)+np.log(priortheta1)+np.log(priortheta2)
        if self.verbose: print('---------- Iter results ----------------')
        Sigma_ = self.create_Gramian(self.X,self.X,self.kernel,theta)
        Sigma_inv_ = pd_inverse(Sigma_)
        f_initial = np.random.multivariate_normal([0]*self.N, self.Sigma).reshape(self.N,1)
        fMAP = scipy.optimize.minimize(lambda f: -self.T(f,theta,Sigma_inv_), f_initial, method='trust-exact',
                       jac=lambda f: -self.T_grad(f,theta,Sigma_inv_), hess=lambda f: -self.T_hessian(f,theta,Sigma_inv_),
                       options={'disp': False,'maxiter': 500}).x
        Lambda_MAP = self.create_Lambda(fMAP,theta[0])
    
        ''' A straightforward (numerically unstable?) implementation '''
        I = np.eye(self.N)
        matrix = I+Sigma_.dot(Lambda_MAP) 
        (sign, logdet) = np.linalg.slogdet(matrix)
        determinant = sign*np.exp(logdet)
        #evidence = np.exp(self.T(fMAP,theta,Sigma_inv_))*np.power(determinant,-0.5)
        log_evidence = self.T(fMAP,theta,Sigma_inv_) - 0.5*np.log(determinant) 
        if self.verbose: print('(scaled) Log-evidence: ' + str(log_evidence))
        if self.verbose: print('Hyper-parameters: ' + str(theta))
        value = log_evidence + prior(theta)
        if np.isnan(value) or not np.isfinite(value):
            if self.verbose: print('Nan log-evidence!')
            return -500
        else:
            if self.verbose: print('(scaled) Log-evidence + Log-prior: ' + str(value))
            return value
        
        ''' An implementation based on Cholesky-decomposition '''
        #try:
        #    posterior_covariance_inv = Sigma_inv_ - Lambda_MAP #Sigma_inv_ + Lambda_MAP
        #    posterior_covariance = pd_inverse(posterior_covariance_inv)
        #    L = np.linalg.cholesky(posterior_covariance)
        #    log_determinant = 2*np.sum(np.log(np.diag(L)))
        #    log_evidence = self.T(fMAP,theta,Sigma_inv_) - 0.5*log_determinant  #determinant still too large, so resulting theta gives too regularized GP
        #    print("Log-determinant: " + str(log_determinant))
        #    print('Hyper-parameters: ' + str(theta))
        #    return log_evidence
        #except:
        #    print('Theta = ' +str(theta) + ' gave a non-PSD covariance matrix.')
        #    return -1e+4

    ''' --- Optimizations --- '''
    
    def update_fMAP(self,random_initial_vector=None,fmap_finding_trials=None):
        ''' Finds MAP estimate and updates it '''
        if fmap_finding_trials is None:
            fmap_finding_trials = self.fMAP_finding_trials  
        if random_initial_vector is None:
            random_initial_vector = self.fMAP_random_initial_vector
        ''' Optimizer: choose second-order either trust-krylov or trust-exact, but trust-exact does not work with pypet multiprocessing! '''   
        if self.fMAP_optimizer=='trust-krylov':
            verbose = False
        else:
            verbose=self.verbose
        if self.verbose: print("MAP-estimation begins...")
        start = time.time()
        min_ = 10**24
        for i in range(0,fmap_finding_trials): #How many times mu_star is tried to find? 
            if self.fMAP is None:
                f_initial = np.random.multivariate_normal([0]*self.N, self.Sigma).reshape(self.N,1)
            elif len(self.fMAP) < self.N and not random_initial_vector:  #Last iteration map estimate is of course a vector of shorter length. Hence add enough constant (=mean of fMAP) to the tail of the map estimate.
                fMAP = np.insert(self.fMAP,len(self.fMAP),[np.mean(self.fMAP)]*(self.N-len(self.fMAP)))
                f_initial = fMAP
            elif len(self.fMAP) == self.N and not random_initial_vector:
                f_initial = self.fMAP
            else:
                f_initial = np.random.multivariate_normal([0]*self.N, self.Sigma).reshape(self.N,1)
            res = scipy.optimize.minimize(lambda f: -self.T(f,self.theta), f_initial, method=self.fMAP_optimizer,
                           jac=lambda f: -self.T_grad(f,self.theta), hess=lambda f: -self.T_hessian(f,self.theta),
                           options={'disp': verbose,'maxiter': self.max_iter_fMAP_estimation})
            if self.verbose: print('... this took ' + str(time.time()-start) + ' seconds.')
            if res.fun < min_:
                min_ = res.fun
                bestfMAP = res.x
        self.fMAP = bestfMAP
    
    def optimize_theta(self):
        ''' Optimizes all hyper-parameters (including noise) by maximizing the evidence '''
        if self.verbose: print("Hyperparameter optimization begins...")
        start = time.time()
        #BayesianOptimization
        #Higher lengthscale generates more accurate GP mean (and location of maximizer of mean)
        #However, higher lengthscale also generates less accurate argmax distribution (argmax samples are widepread)   
        ''' Bounds for hyperparameters: When COVARIANCE_SHRINKAGE = 1e-6'''
        bounds = [{'name': 'sigma', 'type': 'continuous', 'domain': (0.0001,0.1)}, #Too low noise gives non-PSD covariance matrix
                   {'name': 'leghtscale', 'type': 'continuous', 'domain': (0.05,0.9)}, #Theoretical l max: 4*np.sqrt(self.D)
                   {'name': 'sigma_l', 'type': 'continuous', 'domain': (0.01,0.5)}] # since f is a utility function this parameter make no much sense. 
        BO = BayesianOptimization(lambda theta: -self.evidence(theta[0],self.fMAP), #theta[0] since need to unnest list one level
                                  domain=bounds,
                                  optimize_restarts=3,
                                  normalize_Y=True,
                                  initial_design_numdata=20)
        BO.run_optimization(max_iter = 50)
        if self.verbose: print('Optimization of hyperparameters took ' + str(time.time()-start) + ' seconds.')
        self.theta = BO.x_opt
        if self.verbose: print("The optimized theta is "+ str(self.theta))
        
    def mu_star(self,mustar_finding_trials=None):
        ''' Function finds the optimal predictive mean and the maximizer x'''
        if mustar_finding_trials is None:
            mustar_finding_trials = self.mustar_finding_trials
        #Global seach with constraints (DIFFERENTIAL EVOLUTION OPTIMIZATION)
        bounds = self.bounds
        for i in range(0,mustar_finding_trials): #How many times mu_star is tried to find? 
            res = scipy.optimize.differential_evolution(self.mu_pred_neq, bounds,updating='immediate', disp=False,maxiter=2000) 
            #print(res.x) #to monitor how stable are results
            if i==0:
                xstar = res.x
                min_ = res.fun
                xstars_local = res.x
                xstars_local.shape = (1,self.D)
            else:
                if all([bool(np.linalg.norm(x-res.x) > 1e-1) for x in xstars_local]):
                    xstars_local = np.vstack([xstars_local,res.x])
                if res.fun < min_:
                    min_ = res.fun
                    xstar = res.x
        xstar.shape = (self.D,)
        mustar = self.mu_pred(xstar)  
        return xstar,mustar,xstars_local

    ''' --- GP predictions --- '''
    
    def mu_Sigma_pred(self,X_pred):
        ''' Predictive posterior means and covariances '''
        #mean
        k_pred = self.create_Gramian_nonsquare(self.X,X_pred,self.kernel,self.theta)
        mu_pred = k_pred.T.dot(self.Sigma_inv).dot(self.fMAP)
        #covariance
        Sigma_testloc = self.create_Gramian(X_pred,X_pred,self.kernel,self.theta)            
        '''Alternative formula that exploits posterior covariance (this seems to work) '''
        A = self.Sigma_inv - self.Sigma_inv.dot(self.posterior_covariance).dot(self.Sigma_inv) 
        Sigma_pred = Sigma_testloc - k_pred.T.dot(A).dot(k_pred)
        #is_PSD = is_positive_definite(Sigma_pred)
        return mu_pred,Sigma_pred         
    
    def mu_pred(self,X_pred):
        ''' Predict posterior mean for SINGLE vector: needed for optimizers '''
        k_pred = self.create_Gramian_nonsquare(self.X,X_pred.reshape(1,self.D),self.kernel,self.theta)
        mu_pred = k_pred.T.dot(self.Sigma_inv).dot(self.fMAP)
        return float(mu_pred)
    
    def mu_pred_neq(self,X_pred):   
        return -self.mu_pred(X_pred)




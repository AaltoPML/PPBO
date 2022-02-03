import numpy as np
import scipy
import scipy.stats

from scipy.special import ndtr as std_normal_cdf #scipy.special.ndtr fast numerical integration for standard normal cdf

import time

from misc import pd_inverse, var2_normal_pdf


class Hsampler:
    
    def __init__(self, gp_model,nFeatures=1000):
        
        self.nFeatures = nFeatures #Number of random Fourier features
        self.b = None
        self.W = None
        
        self.D = gp_model.D
        self.m = gp_model.m
        self.X = gp_model.X
        self.GP_xstar = gp_model.xstar
        self.GP_xstars_local = gp_model.xstars_local
        self.n_gausshermite_sample_points = gp_model.n_gausshermite_sample_points
        self.obs_indices = gp_model.obs_indices
        self.kernel = str(gp_model.kernel.__name__)
        self.theta = gp_model.theta
        
        self.phi_X = None
        self.omega_MAP = None
        self.covariance = None
        self.covariance_inv = None
        
        self.verbose = False
        
        
    def generate_basis(self):
        ''' Generates random Fourier coefficients for k basis functions '''
        if self.kernel == "SE_kernel":
            l = self.theta[1]
            self.W = np.random.randn(self.nFeatures, self.D) / l
        self.b = np.random.uniform(low=0, high=2*np.pi, size=self.nFeatures)[:,None] 
    
    def phiVec(self,x):
        sigma_f = self.theta[2]  # the kernel amplitude
        return np.sqrt(2.0 * sigma_f**2 / self.nFeatures) * np.cos(np.dot(self.W, x.T) + self.b)
    def phi(self,x):
        sigma_f = self.theta[2]  # the kernel amplitude
        return np.sqrt(2.0 * sigma_f**2 / self.nFeatures) * np.cos(np.dot(self.W, x.T) + self.b.reshape(self.nFeatures,))
    def Dphi(self,x):
        sigma_f = self.theta[2]  # the kernel amplitude
        return -np.sqrt(2.0 * sigma_f**2 / self.nFeatures) * np.dot(np.diag(np.sin(np.dot(self.W, x.T) + self.b.reshape(self.nFeatures,))),self.W)
    def DDphi(self,x):
        raise NotImplementedError
    
    def update_phi_X(self):
        self.phi_X = self.phiVec(self.X)
    
        
    ''' Auxiliary functions for computing transormations/vectroizations of Phi '''    
    def sum_Phi(self,i,order_of_derivative,f,sigma,sample_points=None, weights=None):
        '''
        Auxiliary function that computes summation of Phi-function values
        i = index of x observation, i.e. some element of list 'obs_indices'
        
        dimensionality of the output depends on order_of_derivative
        
        order_of_derivative = how many this the c.d.f of the standard normal (Phi) is differentiated?
        f = vector of f-values
        '''
        m = self.m
        phi_X = self.phi_X 
        #Delta_i_j = (f[i+j+1]-f[i])/(sigma_)
        Delta = f[i+1:i+m+1]
        Delta = (Delta - f[i])/sigma
        if order_of_derivative==0:
            sum_=0
            for j in range(0,m):
                sum_ = sum_ + (1/np.sqrt(np.pi))*np.dot(weights,std_normal_cdf(Delta[j]-np.sqrt(2)*sample_points)) #Do to change of variables to get integteragl into the form int exp(x^2)*....dx. This is why np.sqrt(2)
            return(sum_)
        if order_of_derivative==1:
            sum_=np.zeros(self.nFeatures)
            for j in range(0,m):
                sum_ = sum_ + (phi_X[:,i+1+j]-phi_X[:,i])*float(var2_normal_pdf(Delta[j]))
            return(sum_)
        if order_of_derivative==2:
            sum_=np.zeros(self.nFeatures)
            for j in range(0,m):
                sum_ = sum_ - ((phi_X[:,i+1+j]-phi_X[:,i])**2)*0.5*Delta[j]*float(var2_normal_pdf(Delta[j]))
            return(sum_)
        else:
            print("The derivatives of an order higher than 2 are not needed!")
            return None
        
    def sum_Phi_vec(self,order_of_derivative,f,sigma):
        '''
        Auxiliary function that create a vector of sum_Phi with elements as many as real observations 
        '''
        sample_points, weights = np.polynomial.hermite.hermgauss(self.n_gausshermite_sample_points) #for cross-correlation integral 
        sum_Phi_vec_ = [self.sum_Phi(i,order_of_derivative,f,sigma,sample_points, weights) for i in self.obs_indices]
        return np.array(sum_Phi_vec_)
        
           
    ''' Derivatives of the functional S of the order: 0,1,2 '''
    def S(self,omega,theta):
        f = np.dot(self.phi_X.T,omega)
        sumPhi = self.sum_Phi_vec(0,f,theta[0])
        S = -0.5*omega.T.dot(omega) - np.sum(sumPhi)/self.m
        return S
    
    def S_grad(self,omega,theta):
        f = np.dot(self.phi_X.T,omega)
        sum_Phi_der_vec_ = self.sum_Phi_vec(1,f,theta[0])/(theta[0]*self.m)
        S_grad = -omega - np.sum(sum_Phi_der_vec_,axis=0)      
        return S_grad
    
    def S_hessian(self,omega,theta):
        f = np.dot(self.phi_X.T,omega)
        sum_Phi_der_vec_ = self.sum_Phi_vec(2,f,theta[0])/(self.m*theta[0]**2)
        S_hessian = -np.eye(len(omega)) - np.diag(np.sum(sum_Phi_der_vec_,axis=0)) 
        return S_hessian
    
    def update_omega_MAP(self):
        omega_initial = np.random.randn(self.nFeatures)
        #omega_initial = np.random.normal(0,1,size=self.nFeatures)
        start = time.time()
        res = scipy.optimize.minimize(lambda omega: -self.S(omega,self.theta), omega_initial, method='trust-exact',
                       jac=lambda omega: -self.S_grad(omega,self.theta), hess=lambda omega: -self.S_hessian(omega,self.theta),
                       options={'disp': self.verbose})
        if self.verbose: print('... this took ' + str(time.time()-start) + ' seconds.')
        self.omega_MAP = res.x
        
    def update_covariancematrix(self):
        try:
            self.covariance_inv = -self.S_hessian(self.omega_MAP,self.theta) #self.Sigma_inv + self.Lambda_MAP
            self.covariance = pd_inverse(self.covariance_inv)
        except:
            print('---!!!--- Posterior covariance matrix is not PSD ---!!!---')
            pass
        
        
    def return_xstar(self,omega):
        start = time.time()
        min_trials = 5
        max_trials = 30
        fval = -1e+10
        xstar = None
        i=0
        while xstar is None or i<min_trials:
            if i > max_trials:
                print('Bad omega sample: unable to find f_approx maximizer')
                break
            i+=1
            
            ''' Option 1 '''
            #x_initial = np.random.uniform(0,1,size=self.D)
            ''' Option 2 '''
            #Best initial guess is the maximizer of the posterior mean (plus perturbation)
            #x_initial = np.clip(self.GP_xstar + 0.01*np.random.uniform(0,1,size=self.D),0,1)
            ''' Option 3 '''
            #Pick random local maxima (plus perturbation) as initial value
            x_initial = self.GP_xstars_local[np.random.randint(self.GP_xstars_local.shape[0])]
            x_initial = np.clip(x_initial + 0.01*np.random.uniform(0,1,size=self.D),0,1)
            
            res = scipy.optimize.minimize(lambda x: -np.dot(self.phi(x).T,omega), x0=x_initial, method='L-BFGS-B',bounds=((0,1),)*self.D,
                                          jac=lambda x: -np.dot(self.Dphi(x).T,omega), #hess=lambda x: -np.dot(self.DDphi(x).T,omega),
                                          options={'disp': False,'maxiter': 5000}) #'maxiter': 5000
            xcandidate = res.x
            fval_ = np.dot(self.phi(xcandidate).T,omega)
            if fval_ > fval and all([xcandidate[d]>=0 and xcandidate[d]<=1 for d in range(self.D)]): #Check is there improvement and candidate within boundaries
                fval = fval_
                xstar = xcandidate
                
        if self.verbose: print('Optimization of f_approx took ' + str(time.time()-start) + ' seconds.')
        return xstar
    
    
    ''' Additional functionality: Function for sampling best value for dim given values of other dims (ref_vec) '''
    def return_xstar_for_dim(self,omega,dim,x_ref):
        start = time.time()
        min_trials = 5
        max_trials = 30
        fval = -1e+10
        xstar = None
        i=0
        def x(x_dim):
            x_ref[dim-1]=x_dim[0]
            return x_ref
        while xstar is None or i<min_trials:
            if i > max_trials:
                print('Bad omega sample: unable to find f_approx maximizer')
                break
            i+=1
            x_initial = np.array(self.GP_xstar[dim-1])
            res = scipy.optimize.minimize(lambda x_dim: -np.dot(self.phi(x(x_dim)).T,omega), x0=x_initial, method='Nelder-Mead',bounds=((0,1),),
                                          options={'disp': False,'maxiter': 5000}) 
            xcandidate = x(res.x)
            fval_ = np.dot(self.phi(xcandidate).T,omega)
            if fval_ > fval and all([xcandidate[d]>=0 and xcandidate[d]<=1 for d in range(self.D)]): #Check is there improvement and candidate within boundaries
                fval = fval_
                xstar = xcandidate 
        if self.verbose: print('Optimization of f_approx took ' + str(time.time()-start) + ' seconds.')
        return xstar
    
    
    def sample_omega(self):
        try:
            omega = np.random.multivariate_normal(self.omega_MAP,self.covariance)
        except:
            omega = self.omega_MAP
            print("Omega sampler error! Omega MAP-estimate was used instead.")
        return omega
    
    def sample_xstar(self):
        xstar = None
        while xstar is None:
            omega = self.sample_omega()
            xstar = self.return_xstar(omega)     
        return xstar
    
    def sample_xstar_for_dim(self,dim,x_ref):
        xstar = None
        while xstar is None:
            omega = self.sample_omega()
            xstar = self.return_xstar_for_dim(omega,dim,x_ref)     
        return xstar

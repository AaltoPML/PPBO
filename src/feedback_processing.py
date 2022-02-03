import numpy as np
import scipy
import scipy.stats

from TGN_distribution import TGN_sample
from misc import alpha_bounds


class FeedbackProcessing:
    """
    Class for handling feedback data. Particularly, the class manages the creation of the design matrix X.
    """

    def __init__(self,D,m,original_bounds,alpha_grid_distribution,TGN_speed):
        """
        Initializes the Feedback_Processing object
        """
        self.D = D   #Problem dimension
        self.m = m #How many pseudo-observations per an observation?
        self.original_bounds = original_bounds #Boundaries of each variables as a sequence of tuplets
        self.bounds = ((0, 1),) * self.D
        self.alpha_grid_distribution = alpha_grid_distribution
        self.TGN_speed = TGN_speed
        self.iter_number = 1
        
        self.X_obs = None
        self.X_full = None #Design matrix plus two additional columns
        self.X = None   #Design matrix
        self.N = None   #Number of observations including pseudo-observations
        self.obs_indices = None  #Locations of true observations
        self.pseudobs_indices = None  #Locations of pseudo-observations
        self.latest_obs_indices = None  #Location of latest true observation given all observations from 0 to N
        
    
    ''' Wrapper function '''
    def initialize_data(self,X_obs):
        self.X_obs = X_obs
        self.create_X()
        self.create_indices_bookkeeping() 
    def update_data(self,X_obs):
        self.iter_number = self.iter_number + 1 #Every time data is update increase the iteration count
        self.X_obs = X_obs
        self.update_X()
        self.create_indices_bookkeeping() 
    
        
    def xi_grid(self,xi,x=None,alpha_grid_distribution=None,alpha_star=None,m=None,is_scaled=False):
        if alpha_grid_distribution is None:
            alpha_grid_distribution = self.alpha_grid_distribution
        if m is None:
            m = self.m
            
        '''
        Creates a grid of points for the dimension d.
        If concenrated_on_feedback is selected, then also a point alpha_star should be provided. Grid points will be concentrated around that point.
        ''' 
        if is_scaled:
                alpha_min = 0
                alpha_max = 1
        else:
            lower = np.array([i[0] for i in self.original_bounds])
            upper = np.array([i[1] for i in self.original_bounds]) 
            alpha_min,alpha_max = alpha_bounds(xi,lower,upper)
        
  
        if alpha_grid_distribution=='equispaced':
            noise_level_gridpoints = 0.01 #Add random noise to grid points to avoid singluraity issues in datamatrix. Numbers present noise variance as a percentage of the length of the interval of variable bounds: 0.01 is good starting point
            epsilon_boundary = (alpha_max-alpha_min)*(noise_level_gridpoints/2)  #Number away from the boundary
            epsilon_noise = np.abs(alpha_max-alpha_min)*noise_level_gridpoints #Level of noise depends on noise level parameter and length of the interval of var bounds
            alpha = []
            while len(alpha) != m:
                alpha = np.linspace(alpha_min+epsilon_boundary, alpha_max-epsilon_boundary, num=m) + np.random.normal(0, epsilon_noise, m)
                alpha = np.clip(alpha, alpha_min, alpha_max)
                alpha = np.unique(alpha) #delete duplicates   
        elif alpha_grid_distribution=='Cauchy':
            ''' Gridpoints are drawn from Cauchy-dsitribution with location alpha_star '''
            alpha = []
            while len(alpha) != m:
                noise_level_gridpoints = 0.07  #Add random noise to grid points to avoid singluraity issues in datamatrix. Numbers present noise variance as a percentage of the length of the interval of variable bounds: 0.07 is good starting point
                alpha = scipy.stats.cauchy.rvs(loc=float(alpha_star), scale=np.abs(alpha_max-alpha_min)*noise_level_gridpoints, size=m)
                alpha = np.clip(alpha, alpha_min, alpha_max)
                alpha = np.unique(alpha) #delete duplicates
        elif alpha_grid_distribution=='TGN':
            ''' Gridpoints are drawn from Truncated Generalized Normal (TGN) distribution with the location parameter alpha_star and the form parameter gamma '''
            ''' The speed of transformation from uniform distribution to normal distribution as iteration_number ---> infinity '''
            s = self.TGN_speed #speed parameter that lies in (0,1]
            gamma = 3/np.power(np.max([self.iter_number+1-self.D,1]),s) + 2
            
            alpha = []
            while len(alpha) != m:
                alpha = TGN_sample(size=m,gamma=gamma,alpha=float(alpha_star),x_min=alpha_min,x_max=alpha_max)
                alpha = np.clip(alpha, alpha_min, alpha_max)
                alpha = np.unique(alpha) #delete duplicates 
        else:
            print('Uknown alpha-distribution: ' + str(alpha_grid_distribution))
        
        alpha.shape = (m,1)
        xi = np.array(xi).reshape(1,self.D)
        xi_grid = np.matmul(alpha,xi)
        if x is None:
            xi_column_indices = [j for j in range(self.D) if j not in np.where((xi_grid == 0).all(0))[0]]
            #Remove zero columns i.e. columns whrer xi_d = 0 ?
            xi_grid = xi_grid[:,xi_column_indices]
        else:
            # x_column_indices = np.where((xi_grid == 0).all(0))[0]
            #xi_grid[:,x_column_indices] = np.tile(x, (m, 1))
            xi_grid = xi_grid + np.tile(x, (m, 1))
        return xi_grid

    def create_X(self):
        D = self.D
        m = self.m
        X = np.empty((0,2*D + 1))
        for i in range(0,self.X_obs.shape[0]):
            alpha_xi_plus_x = self.X_obs[i,:D]
            xi = self.X_obs[i,D:2*D]
            x = np.zeros((D,))
            x[list(np.where(xi == 0)[0])] = alpha_xi_plus_x[list(np.where(xi == 0)[0])] 
            alpha_star =  self.X_obs[i,-1] #alpha_xi_plus_x[np.where(xi == 1)[0][0]]
            xi_grid = self.xi_grid(xi=xi,x=x,alpha_star=alpha_star)
            matrix = np.vstack([alpha_xi_plus_x,xi_grid])
            matrix = np.hstack([matrix,np.tile(xi.reshape(D,1), (1,m+1)).T]) #what is projection xi column
            matrix = np.hstack([matrix,np.array([0]+[1]*m).reshape(m+1,1)]) #is_pseudobs_row columns
            ''' ready '''
            X = np.concatenate((X,matrix),axis=0)   
        self.X_full = X  #Save X_full matrix
        X = X[:,0:D] #remove xi column and is_pseudobs column
        X = self.scale(X) #scale
        self.X = X
        self.N = len(X)
        
        
    def update_X(self):
        ''' Assuming that X_obs is updated, i.e. one more row added '''
        D = self.D
        m = self.m
        X = self.X_full.copy()
        for i in range(self.X_obs.shape[0]-1,self.X_obs.shape[0]):
            alpha_xi_plus_x = self.X_obs[i,:D]
            xi = self.X_obs[i,D:2*D]
            x = np.zeros((D,))
            x[list(np.where(xi == 0)[0])] = alpha_xi_plus_x[list(np.where(xi == 0)[0])] 
            alpha_star =  self.X_obs[i,-1] #alpha_xi_plus_x[np.where(xi == 1)[0][0]]
            xi_grid = self.xi_grid(xi=xi,x=x,alpha_star=alpha_star)
            matrix = np.vstack([alpha_xi_plus_x,xi_grid])
            matrix = np.hstack([matrix,np.tile(xi.reshape(D,1), (1,m+1)).T]) #what is projection xi column
            matrix = np.hstack([matrix,np.array([0]+[1]*m).reshape(m+1,1)]) #is_pseudobs_row columns
            ''' ready '''
            X = np.concatenate((X,matrix),axis=0)   
        self.X_full = X  #Save X_full matrix
        X = X[:,0:D] #remove xi column and is_pseudobs column
        X = self.scale(X) #scale
        self.X = X
        self.N = len(X)          
        
    
    def is_pseudobs(self,i):
        ''' Returns boolean whether given observation index i is a grid point (or true observation) '''
        is_ = self.X_full[i,2*self.D]
        return bool(is_)
    
    def create_indices_bookkeeping(self):
        self.obs_indices = [i for i in range(0,self.N) if not self.is_pseudobs(i)]
        self.pseudobs_indices = [i for i in range(0,self.N) if self.is_pseudobs(i)]
        self.latest_obs_indices = [np.max([ind for ind in self.obs_indices if ind < i]) if self.is_pseudobs(i) else i for i in range(0,self.N)]
        
    def scale(self,X,retain_0_values=False):
        if retain_0_values:
            A = X.copy()
        bound_mins = np.array([x[0] for x in self.original_bounds])
        bound_maxs = np.array([x[1] for x in self.original_bounds])
        X = X - bound_mins
        X = X / np.abs(bound_maxs - bound_mins)
        if retain_0_values:
            X[A==0] = 0
        return X
    
    def unscale(self,X,retain_0_values=False):
        if retain_0_values:
            A = X.copy()
        bound_mins = np.array([x[0] for x in self.original_bounds])
        bound_maxs = np.array([x[1] for x in self.original_bounds])
        X = X*np.abs(bound_maxs - bound_mins)
        X = X + bound_mins
        if retain_0_values:
            X[A==0] = 0
        return X
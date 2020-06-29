

class PPBO_settings:
    """
    Class that specifies the settings for Projective Preferential Bayesian Optimization
    """

    def __init__(self,D,bounds,xi_acquisition_function,theta_initial=[0.001,20,0.001],
                 user_feedback_grid_size=100,m=25,verbose=False,EI_EXR_mc_samples=200,EI_EXR_BO_maxiter=30):
        
        """
        BASIC SETTINGS
        """
        self.verbose = verbose #Wheter or not to print what is happening under the hood
        self.user_feedback_grid_size = user_feedback_grid_size
        self.n_gausshermite_sample_points = 30 #How many sample points in GaussHermite quadrature for approximating the convolution in the likelihood?

        """
        THE DOMAIN OF THE PROBLEM 
        """
        self.D = D   #Problem dimension
        self.original_bounds = bounds   #((xmin,xmax),)*self.D #Boundaries of each variables as a sequence of tuplets
        
        """
        SETTINGS FOR THE OPTIMIZERS 
        """
        self.max_iter_fMAP_estimation = 500
        self.fMAP_optimizer = 'trust-exact'   #scipy optimizer for f_MAP-estimation: trust-krylov or trust-exact
        self.mu_star_finding_trials = 1 #This depends on the noise level. #Good default values is 1.
         
        '''HYPERPARAMETER INITIAL VALUES'''
        self.theta_initial = theta_initial #Intial hyperparameters. Put None if you want keep default hyper-parameters.
       
        ''' PSEUDO-OBSERVATIONS '''
        self.n_pseudoobservations = m  #How many pseudoobservations. Note that Sigma condition number grows w.r.t. that!
        self.alpha_grid_distribution = 'TGN'   #evenly, cauchy or TGN (truncated generalized normal distribution)
        self.TGN_speed = 0.4 #a speed of transformation from uniform dist to normal dist if TGN is selected,  0.3-0.4
        
        ''' ACQUISITION STRATEGY '''
        #PCD,EI,EXT,EXR, or RAND:
        self.xi_acquisition_function = xi_acquisition_function         
        if self.xi_acquisition_function == "EI" or self.xi_acquisition_function == "EXR":
            self.x_acquisition_function = 'none'
            #Initial nonzero dimensions of xi
            self.xi_dims_prev_iter = list(range(self.D - 1))
            #self.xi_dims_prev_iter = [0,1]
            self.mc_samples = EI_EXR_mc_samples
            self.BO_maxiter = EI_EXR_BO_maxiter        
        elif self.xi_acquisition_function == "PCD" or self.xi_acquisition_function == "EXT" :
            self.dim_query_prev_iter = self.D #That is, PCD/EXT starts from 1st dimension
            self.x_acquisition_function = 'exploit'
        elif self.xi_acquisition_function == "RAND":
            self.x_acquisition_function = 'random'
        else:
            print('Unknown acquisition strategy!')
            
        '''Want override x_acquisition strategy? '''
        #self.x_acquisition_function = 'xxx'




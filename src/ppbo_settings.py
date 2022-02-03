

class PPBO_settings:
    """
    Class that specifies the settings for Projective Preferential Bayesian Optimization
    """

    def __init__(self,
                 D,
                 bounds,
                 xi_acquisition_function,
                 theta_initial= [1,0.1,8],
                 user_feedback_grid_size=100,  
                 m=25,
                 verbose=True,
                 EI_EXR_mc_samples=150,        
                 EI_EXR_BO_maxiter=20,
                 mustar_finding_trials=3, 
                 kernel='SE_kernel',
                 skip_computations_during_initialization=True, #Remember GP_model.turn_initialization_off() in PPBO-loop after final initial query!,
                 skip_xstaroptimization_during_initialization=False,
                 alpha_grid_distribution='equispaced'):         
        
        """
        BASIC SETTINGS
        """
        self.verbose = verbose #Wheter or not to print what is happening under the hood
        self.user_feedback_grid_size = user_feedback_grid_size
        self.skip_computations_during_initialization=skip_computations_during_initialization
        self.skip_xstaroptimization_during_initialization=skip_xstaroptimization_during_initialization

        """
        THE DOMAIN OF THE PROBLEM 
        """
        self.D = D   #Problem dimension
        self.original_bounds = bounds   #((xmin,xmax),)*self.D, Boundaries of each variables as a sequence of tuplets
        
        """
        SETTINGS FOR THE OPTIMIZERS 
        """
        self.fMAP_optimizer = 'trust-exact'   #scipy optimizer for f_MAP-estimation: trust-krylov or trust-exact
        self.mustar_finding_trials = mustar_finding_trials  #This depends on the noise level. Good default values is 4.
         
        '''KERNEL AND HYPERPARAMETER INITIAL VALUES'''
        self.kernel = kernel #Kernel type as a string
        self.theta_initial = theta_initial #Intial hyperparameters. Put None if you want keep default hyperparameters.
       
        ''' PSEUDO-OBSERVATIONS '''
        self.n_pseudoobservations = m  #How many pseudo-observations. Note that Sigma condition number grows w.r.t. that!
        self.alpha_grid_distribution = alpha_grid_distribution  #How pseudo-observations are distributed?: equispaced, Cauchy or TGN (truncated generalized normal distribution). Default: equispaced
        self.TGN_speed = 0.4 #a speed of transformation from uniform dist to normal dist if TGN is selected,  0.3-0.4. Note this the higher self.TGN_speed, the slower is the transformation.
        self.n_gausshermite_sample_points = 200 #Old = 40 #How many sample points in GaussHermite quadrature for approximating the convolution in the likelihood?

        ''' ACQUISITION STRATEGY '''
        #Strategies available: [PCD,EXT,RAND,EI,EI-FIXEDX,EXR,EI-EXT,EI-EXT-FAST,EI-VARMAX,EI-VARMAX-FAST,COORDINATE-VARMAX]
        self.mc_samples = EI_EXR_mc_samples
        self.BO_maxiter = EI_EXR_BO_maxiter
        self.xi_acquisition_function = xi_acquisition_function
        if self.xi_acquisition_function == "PCD" or self.xi_acquisition_function == "EXT":
            self.dim_query_prev_iter = self.D #That is, PCD/EXT starts from 1st dimension
            self.x_acquisition_function = 'exploit'
        elif self.xi_acquisition_function == "RAND":
            self.x_acquisition_function = 'random'
        elif self.xi_acquisition_function == 'EI' or self.xi_acquisition_function == 'EI-FIXEDX' or self.xi_acquisition_function == 'EXR':
            self.x_acquisition_function = 'none'
            #Initial nonzero dimensions of xi
            if self.D > 2:
                self.xi_dims_prev_iter = [0,1]
            else:
                self.xi_dims_prev_iter = [1]
            #self.xi_dims_prev_iter = list(range(self.D - 1))
        elif self.xi_acquisition_function == 'EI-EXT' or self.xi_acquisition_function == 'EI-EXT-FAST':
            self.x_acquisition_function = 'exploit'
        elif self.xi_acquisition_function == 'EI-VARMAX' or self.xi_acquisition_function == 'EI-VARMAX-FAST' or self.xi_acquisition_function == 'COORDINATE-VARMAX':
            self.x_acquisition_function = 'varmax'
            if self.xi_acquisition_function == 'COORDINATE-VARMAX':
                self.dim_query_prev_iter = self.D #That is, PCD/EXT starts from 1st dimension
        else:
            print("Unknown acquisition function!")




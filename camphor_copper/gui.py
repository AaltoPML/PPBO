import os
import sys
from datetime import datetime

import ase
import ase.visualize
import nglview
import ipywidgets as widgets
import pandas as pd
import numpy as np

from create_111_camphor_func import create_file
from create_111_camphor_func import create_geometry
from feedback_processing import FeedbackProcessing

path_from_root_to_files = os.getcwd() + '/camphor_copper/'


class GUI_session:
    """
    Class for handling Graphical User Inteterface:
    """
    def __init__(self,PPBO_settings):
        """
        Basic settings
        """
        
        self.D = PPBO_settings.D   #Problem dimension
        self.bounds = PPBO_settings.original_bounds #Boundaries of each variables as a sequence of tuplets
        self.alpha_grid_distribution = PPBO_settings.alpha_grid_distribution
        
        self.preference_feedback_size = 100 #How many possible points in the user feedback grid?
        self.confidence_feedback_size = 5 #How many possible points in the user feedback grid?
        self.FP = FeedbackProcessing(self.D, self.preference_feedback_size,self.bounds,self.alpha_grid_distribution,PPBO_settings.TGN_speed)
              
        self.current_xi = None
        self.current_x = None
        self.current_xi_grid = None
        self.user_feedback_preference = None  #variable to store user feedback for preference
        self.user_feedback_confidence = None  #variable to store user feedback for confidence degree
        self.user_feedback_was_given = False  
        self.popup_configuration_movie_has_been_called = False
        self.movie = None
        
        self.results_preference = pd.DataFrame(columns=(['alpha_xi_x' + str(i) for i in range(1,6+1)] 
                    + ['xi' + str(i) for i in range(1,6+1)]
                    + ['alpha_star']),dtype=np.float64)  #The output of the session is a dataframe containing user feedback
        self.results_confidence = pd.DataFrame(columns=(['x' + str(i) for i in range(1,6+1)] 
                    + ['xi' + str(i) for i in range(1,6+1)]
                    + ['confidence']),dtype=np.float64)  #The output of the session is a dataframe containing user feedback
        
    
    ''' Auxiliary functions '''    
    def set_xi(self,xi):
        self.current_xi = xi
    def set_x(self,x):
        self.current_x = x
    def create_xi_grid(self):
        self.current_xi_grid = self.FP.xi_grid(xi=self.current_xi,x=self.current_x,
                                               alpha_grid_distribution='equispaced',
                                               alpha_star=None,m=self.preference_feedback_size,
                                               is_scaled=False)
 

    
    def create_movie_of_configuration(self):
        """
        Create 'movie.traj'.
        """   
        trajectory = []
        for i in range(self.current_xi_grid.shape[0]):
            conf = np.array([list(self.current_xi_grid[i,:])])
            function_arguments = pd.DataFrame(data=conf,
                                              columns=['camp_dx','camp_dy','camp_origin_height','alpha','beta','gamma'])
            function_arguments = function_arguments.to_dict('records')[0]
            trajectory.append(create_geometry(**function_arguments))      
        ase.io.write(path_from_root_to_files+'movie.traj',images=trajectory)
        movie = ase.io.read(path_from_root_to_files+'movie.traj', index=':')
        self.movie = movie
    
    def getMiniGUI(self):
        view = nglview.show_asetraj(self.movie)
        view.parameters = dict(background_color='white',camera_type='perpective',camera_fov=15)
        view._camera_orientation = [-28.583735327243016, -0.2970873285220947, 1.198387795047608, 0, -0.3455812695981218, 28.584920668432527, -1.1563751171127739, 0, -1.1853133653976955, -1.1697730312356562, -28.561879887836003, 0, -7.061999797821045, -8.524999618530273, -8.855999946594238, 1] #Default camera view
        button = widgets.Button(description='Confirm',disabled=False,button_style='')
        slider = widgets.IntSlider(min=0, max=self.confidence_feedback_size-1, step=1, description='Confidence: ', 
                                   value=0,continuous_update=False, readout=True,layout=widgets.Layout(width='60%', height='80px',position='right'))
        def confirm(event):
            pref_feedback = int(view.frame)
            conf_feedback = int(slider.value)
            self.user_feedback_preference = self.current_xi_grid[(pref_feedback),:]
            self.user_feedback_confidence = conf_feedback
            self.user_feedback_was_given = True                
        button.on_click(confirm)
        return view,button,slider
        
    def save_results(self):
        res_preference = pd.DataFrame(columns=(['alpha_xi_x' + str(i) for i in range(1,6+1)] 
                    + ['xi' + str(i) for i in range(1,6+1)]
                    + ['alpha_star']),dtype=np.float64)
        res_confidence = pd.DataFrame(columns=(['x' + str(i) for i in range(1,6+1)] 
                    + ['xi' + str(i) for i in range(1,6+1)]
                    + ['confidence']),dtype=np.float64)    
        xi = self.current_xi
        x = self.current_x
        alpha_xi_x_preference = self.user_feedback_preference
        confidence = self.user_feedback_confidence
        alpha_star_preference = np.nanmin(alpha_xi_x_preference[x==0]/xi[x==0])  #every component in alpha_xi_x[x==0]/xi[x==0] should be same
        new_row_preference = list(alpha_xi_x_preference) + list(xi) + [alpha_star_preference]
        new_row_confidence = list(x) + list(xi) + [confidence]
        res_preference.loc[0,:] = new_row_preference
        res_confidence.loc[0,:] = new_row_confidence
        self.results_preference=self.results_preference.append(res_preference, ignore_index=True)
        self.results_confidence=self.results_confidence.append(res_confidence, ignore_index=True)

    def initialize_iteration(self,x,xi):
        self.set_x(x)
        self.set_xi(xi)
        self.create_xi_grid()
        self.create_movie_of_configuration()
        

def generate_optimal_configuration(x_star_unscaled):
    dict_x_star = dict(zip(['camp_dx','camp_dy','camp_origin_height','alpha','beta','gamma'],x_star_unscaled))
    print('The optimal configuration: ' + str(dict_x_star))
    create_file(**dict_x_star) #x_star
    system = ase.io.read(path_from_root_to_files+'geometry.in')
    HTML = ase.visualize.view(system, viewer="x3d").data   
    filename = 'optimal_x_'+str(datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))+'.html'
    file = open(path_from_root_to_files+filename,'w') 
    file.write(HTML) 
    file.close()
    return filename
	

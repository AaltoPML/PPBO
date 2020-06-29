import os
import time
from datetime import datetime
import psutil #to kill python processes i.e. close windows

import ase
import ase.visualize
import ase.gui.gui  

import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import ttk

from Camphor_Copper.create_111_camphor_func import create_file
from Camphor_Copper.create_111_camphor_func import create_geometry
from feedback_processing import FeedbackProcessing

path_from_root_to_files = os.getcwd() + '/Camphor_Copper/'


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
        
        self.user_feedback_grid_size = 100 #How many possible points in the user feedback grid?
        self.FP = FeedbackProcessing(self.D, self.user_feedback_grid_size,self.bounds,self.alpha_grid_distribution,PPBO_settings.TGN_speed)
              
        self.current_xi = None
        self.current_x = None
        self.current_xi_grid = None
        self.user_feedback = None  #variable to store user feedback
        self.user_feedback_was_given = False  
        self.popup_configuration_movie_has_been_called = False
        
        self.results = pd.DataFrame(columns=(['alpha_xi_x' + str(i) for i in range(1,6+1)] 
                    + ['xi' + str(i) for i in range(1,6+1)]
                    + ['alpha_star']),dtype=np.float64)  #The output of the session is a dataframe containing user feedback
        
    
    ''' Auxiliary functions '''    
    def set_xi(self,xi):
        self.current_xi = xi
    def set_x(self,x):
        self.current_x = x
    def create_xi_grid(self):
        self.current_xi_grid = self.FP.xi_grid(xi=self.current_xi,x=self.current_x,
                                               alpha_grid_distribution='evenly',
                                               alpha_star=None,m=self.user_feedback_grid_size,
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
            
       
    def popup_configuration_movie(self):
       
        if not self.popup_configuration_movie_has_been_called:
            movie = ase.io.read(path_from_root_to_files+'movie.traj', index=':')
            ase.visualize.view(movie)
        else:
            PROCNAME = "python" #AD-HOC SOLUTION TO SHUT DOWN PREVIOUS GUI WINDOW!!!!!
            for proc in psutil.process_iter():
                if proc.name() == PROCNAME:
                    proc.kill()
            movie = ase.io.read(path_from_root_to_files+'movie.traj', index=':')
            ase.visualize.view(movie)
        self.popup_configuration_movie_has_been_called = True
     
    def popup_message(self):
        popup = tk.Tk()
        popup.attributes('-topmost', True) #The message is always on the top
        #popup.wm_title("Which configuration you expect to achieve the lowest energy?")
        popup.wm_title(" ")
        label = ttk.Label(popup, text="Type number: ", 
                           font=("Helvetica", 14))
        E1 = ttk.Entry(popup, text="")
        B2 = ttk.Button(popup, text="Confirm", command = lambda: self.buttom_action("confirm",popup,E1)) 
        B3 = ttk.Button(popup, text="I dont't know.", command = lambda: self.buttom_action("dont_know",popup,E1)) 
        label.pack(side="top", fill="x", pady=10)    
        E1.pack()
        B2.pack()
        B3.pack()
        popup.mainloop()   

    def buttom_action(self,user_input,popup,E1):
        """
        Decides what happens when one of three buttons is clicked
        """
        if user_input=="confirm":
            typed_value = float(E1.get())
            if typed_value>self.user_feedback_grid_size or typed_value < 1:
                print("Invalid input!")
                typed_value = 1
            self.user_feedback = self.current_xi_grid[(int(typed_value)-1),:] #i.e. typed value - 1 !!
            print("--- Feedback ---")
            print("Typed value: " + str(typed_value))
            print("... converted to: " + str(self.user_feedback))
            self.user_feedback_was_given = True
        elif user_input=="dont_know":
            raise NotImplementedError
        else:
            print("Error, something strange happened!")
    
        popup.destroy()  #Shut down tkinter.Tk() instance 
        time.sleep(0.2)
        
    def save_results(self):
        res = pd.DataFrame(columns=(['alpha_xi_x' + str(i) for i in range(1,6+1)] 
                    + ['xi' + str(i) for i in range(1,6+1)]
                    + ['alpha_star']),dtype=np.float64)    
        xi = self.current_xi
        x = self.current_x
        alpha_xi_x = self.user_feedback
        alpha_star = np.nanmin(alpha_xi_x[x==0]/xi[x==0])  #every component in alpha_xi_x[x==0]/xi[x==0] should be same
        new_row = list(alpha_xi_x) + list(xi) + [alpha_star]
        res.loc[0,:] = new_row
        self.results=self.results.append(res, ignore_index=True)

    def run_iteration(self,allow_feedback):
        """
        Runs one iteration of the session.
        Makes sure that configurations are set correctly.
        """
        self.create_xi_grid()
        if allow_feedback:
            while not self.user_feedback_was_given:
                self.create_movie_of_configuration()
                self.popup_configuration_movie()
                self.popup_message()
                print("Iteration done!")
            self.user_feedback_was_given = False
            self.save_results()
        

def generate_optimal_configuration(x_star_unscaled):
    dict_x_star = dict(zip(['camp_dx','camp_dy','camp_origin_height','alpha','beta','gamma'],x_star_unscaled))
    print('The optimal configuration: ' + str(dict_x_star))
    create_file(**dict_x_star) #x_star
    system = ase.io.read(path_from_root_to_files+'geometry.in')
    HTML = ase.visualize.view(system, viewer="x3d").data   
    filename = path_from_root_to_files+'optimal_x_'+str(datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))+'.html'
    file = open(filename,'w') 
    file.write(HTML) 
    file.close()
    return filename
	

#!/usr/bin/env python
#Petrus Mikkola (2020) @ Aalto University, Espoo, Finland


#To solve issues with openBLAS, MKL and Python multiprocessing (https://stackoverflow.com/questions/17053671/python-how-do-you-stop-numpy-from-multithreading)
#set the following environmental variables BEFORE opening Python
#export MKL_NUM_THREADS=1
#export OPENBLAS_NUM_THREADS=1
#export OMP_NUM_THREADS=1
#export NUMEXPR_NUM_THREADS=1
import os
#os.environ.update(
#    OMP_NUM_THREADS = '1',
#    OPENBLAS_NUM_THREADS = '1',
#    NUMEXPR_NUM_THREADS = '1',
#    MKL_NUM_THREADS = '1',
#)

import sys
#Set the correct working directory
wd = '/u/18/mikkolp2/unix/Desktop/PPBO/'
os.chdir(wd)

import pandas as pd
import numpy as np
import time
from datetime import datetime

from ppbo_settings import PPBO_settings
from gp_model import GPModel
from acquisition import next_query
from PPBO_numerical.test_functions import pp_ackley, pp_hartmann6d, pp_levy, pp_sixhump_camel, hartmann6d_orig, ackley_orig, levy_orig, sixhump_camel_orig

from pypet import Environment



def make_query(objective,xi,x):
    if objective=='six_hump_camel':
        alpha_star = pp_sixhump_camel(xi,x)
    if objective=='levy':
        alpha_star = pp_levy(xi,x)
    if objective=='ackley':
        alpha_star = pp_ackley(xi,x)
    if objective=='hartmann6d':
        alpha_star = pp_hartmann6d(xi,x)
    print("Query result alpha_star= "+ str(alpha_star))
    print("Query result alpha_star*xi + x= "+ str(alpha_star*xi + x))
    return alpha_star
    
def evaluate_objective(objective,x):
    if objective=='six_hump_camel':
        value = sixhump_camel_orig(x)
    if objective=='levy':
        value = levy_orig(x)
    if objective=='ackley':
        value = ackley_orig(x)
    if objective=='hartmann6d':
        value = hartmann6d_orig(x)
    return value


def run_ppbo_loop(objective,initial_queries_xi,initial_queries_x,number_of_actual_queries,PPBO_settings):
    
    ''' ----------------- MAIN LOOP ------------------------- '''
    
    
    mu_star_results = [0]*(number_of_actual_queries+len(initial_queries_xi))
    x_star_results = np.empty([number_of_actual_queries+len(initial_queries_xi),PPBO_settings.D])
    results = pd.DataFrame(columns=(['alpha_xi_x' + str(i) for i in range(1,PPBO_settings.D+1)] 
    + ['xi' + str(i) for i in range(1,PPBO_settings.D+1)]
    + ['alpha_star']),dtype=np.float64)
    def save_results(old_results,alpha_xi_x,xi,alpha_star):
        new_row = list(alpha_xi_x) + list(xi) + [alpha_star]
        res=np.vstack([old_results, new_row])
        return res

    ''' Initialial queries -loop'''
    for i in range(len(initial_queries_xi)):
        #Set query location
        if not i==0 and ADAPTIVE_INITIALIZATION: 
            initial_queries_x[i:,:] = alpha_star*xi + x
        
        x = np.array(initial_queries_x[i])
        xi = np.array(initial_queries_xi[i])
        x[xi!=0] = 0

        ''' Query '''
        alpha_star = make_query(objective,xi,x)
        results = save_results(results,alpha_star*xi + x,xi,alpha_star)
        
        ''' Create and update GP-model first time '''
        if i==0:
            GP_model = GPModel(PPBO_settings)               
        GP_model.update_feedback_processing_object(np.array(results))
        GP_model.update_data()
        GP_model.update_model()
        x_star_ = GP_model.FP.unscale(GP_model.x_star_)
        mu_star_results[i] = GP_model.mu_star_
        x_star_results[i,:] = x_star_
        print("x_star of the initialization " + str(i+1)+"/"+str(len(initial_queries_xi))+ ' is ' + str(x_star_))
        
    if OPTIMIZE_HYPERPARAMETERS_AFTER_INITIALIZATION:
        GP_model.update_model(optimize_theta=OPTIMIZE_HYPERPARAMETERS_AFTER_INITIALIZATION)  
    
    print('Initialization done! (Acq.' +str(PPBO_settings.xi_acquisition_function)+' )')
    
    ''' Queries -loop '''
    for i in range(number_of_actual_queries):
        print("Starting query " + str(i+1)+"/"+str(number_of_actual_queries)+" ...")
        
        ''' Compute next query '''
        xi_next,x_next = next_query(PPBO_settings,GP_model,unscale=True)
        
        ''' Query '''
        alpha_star = make_query(objective,xi_next,x_next)
        results = save_results(results,alpha_star*xi_next + x_next,xi_next,alpha_star)
        
        ''' Save results '''
        GP_model.update_feedback_processing_object(np.array(results))
        GP_model.mu_star_previous_iteration = GP_model.mu_star_#GP_model.mu_star()[1]
        ''' Update the model '''
        GP_model.update_data()
        if i+1==OPTIMIZE_HYPERPARAMETERS_AFTER_ACTUAL_QUERY_NUMBER:
            GP_model.update_model(optimize_theta=True)     
        else:
            GP_model.update_model(optimize_theta=OPTIMIZE_HYPERPARAMETERS_AFTER_EACH_ITERATION)
        x_star_ = GP_model.FP.unscale(GP_model.x_star_)
        print("x_star of the iteration: " + str(x_star_))
        print("The objective value at that point: " + str(-evaluate_objective(objective,x_star_)))
        mu_star_results[len(initial_queries_xi)+i] = GP_model.mu_star_
        x_star_results[len(initial_queries_xi)+i,:] = x_star_
    
    print('Run done! (Acq.' +str(PPBO_settings.xi_acquisition_function)+' )')

    return results, x_star_results, mu_star_results, GP_model




def six_hump_camel(traj):
    PPBO_settings_ = PPBO_settings(D=2,bounds=((-3,3),(-2,2)),
                                   xi_acquisition_function=traj.xi_acquisition_function,m=traj.m,
                                   theta_initial=[0.001,200,0.01]) 
    initial_queries_xi = np.diag([PPBO_settings_.original_bounds[i][1] for i in range(PPBO_settings_.D)])
    np.random.seed(traj.initialization_seed) 
    initial_queries_x = np.random.uniform([PPBO_settings_.original_bounds[i][0] for i in range(PPBO_settings_.D)], [PPBO_settings_.original_bounds[i][1] for i in range(PPBO_settings_.D)], (len(initial_queries_xi), PPBO_settings_.D))
    results,x_star_results,mu_star_results,GP_model = run_ppbo_loop('six_hump_camel',initial_queries_xi,initial_queries_x,traj.number_of_actual_queries,PPBO_settings_)
    traj.f_add_result('x_star',x_star_results)
    traj.f_add_result('mu_star',mu_star_results)
    return GP_model


def levy(traj):
    D = 10
    PPBO_settings_ = PPBO_settings(D=D,bounds=((-10,10),)*D,
                                   xi_acquisition_function=traj.xi_acquisition_function,m=traj.m,
                                   theta_initial=[0.001,5,0.01])
    initial_queries_xi = np.diag([PPBO_settings_.original_bounds[i][1] for i in range(PPBO_settings_.D)])
    np.random.seed(traj.initialization_seed) 
    initial_queries_x = np.random.uniform([PPBO_settings_.original_bounds[i][0] for i in range(PPBO_settings_.D)], [PPBO_settings_.original_bounds[i][1] for i in range(PPBO_settings_.D)], (len(initial_queries_xi), PPBO_settings_.D))
    results,x_star_results,mu_star_results,GP_model= run_ppbo_loop('levy',initial_queries_xi,initial_queries_x,traj.number_of_actual_queries,PPBO_settings_)
    traj.f_add_result('x_star',x_star_results)
    traj.f_add_result('mu_star',mu_star_results)
    return GP_model

def ackley(traj):
    D = 20
    PPBO_settings_ = PPBO_settings(D=D,bounds=((-32.768, 32.768),)*D,
                                      xi_acquisition_function=traj.xi_acquisition_function,m=traj.m,
                                      theta_initial=[0.001,4,0.01]) 
    initial_queries_xi = np.diag([PPBO_settings_.original_bounds[i][1] for i in range(PPBO_settings_.D)])
    np.random.seed(traj.initialization_seed) 
    initial_queries_x = np.random.uniform([PPBO_settings_.original_bounds[i][0] for i in range(PPBO_settings_.D)], [PPBO_settings_.original_bounds[i][1] for i in range(PPBO_settings_.D)], (len(initial_queries_xi), PPBO_settings_.D))
    results,x_star_results,mu_star_results,GP_model = run_ppbo_loop('ackley',initial_queries_xi,initial_queries_x,traj.number_of_actual_queries,PPBO_settings_)
    traj.f_add_result('x_star',x_star_results)
    traj.f_add_result('mu_star',mu_star_results)
    return GP_model

def hartmann6d(traj):
    PPBO_settings_ = PPBO_settings(D=6,bounds=((0, 1),)*6,
                                   xi_acquisition_function=traj.xi_acquisition_function,m=traj.m,
                                   theta_initial=[0.001,5,0.01]) 
    initial_queries_xi = np.eye(PPBO_settings_.D)
    np.random.seed(traj.initialization_seed) 
    initial_queries_x = np.random.uniform([PPBO_settings_.original_bounds[i][0] for i in range(PPBO_settings_.D)], [PPBO_settings_.original_bounds[i][1] for i in range(PPBO_settings_.D)], (len(initial_queries_xi), PPBO_settings_.D))
    results,x_star_results,mu_star_results,GP_model = run_ppbo_loop('hartmann6d',initial_queries_xi,initial_queries_x,traj.number_of_actual_queries,PPBO_settings_)
    traj.f_add_result('x_star',x_star_results)
    traj.f_add_result('mu_star',mu_star_results)
    return GP_model

''' Run experimetns '''
NUMBER_OF_QUERIES = 98
ADAPTIVE_INITIALIZATION = False #Set previous feedback for d, for value of d-th-coordinate of intial query
OPTIMIZE_HYPERPARAMETERS_AFTER_INITIALIZATION = False
OPTIMIZE_HYPERPARAMETERS_AFTER_EACH_ITERATION = False
OPTIMIZE_HYPERPARAMETERS_AFTER_ACTUAL_QUERY_NUMBER = 999

env = Environment(trajectory='numerical_experiments_trajectory_'+str(datetime.now().strftime("%d-%m-%Y_%H-%M-%S")),overwrite_file=True,
                  multiproc=True,ncores=0,use_pool=False,wrap_mode='LOCAL')
traj = env.trajectory
traj.f_add_parameter('initialization_seed',0,comment='Seed for sampling initial points')
traj.f_add_parameter('number_of_actual_queries',NUMBER_OF_QUERIES,comment='Keep this constant')
traj.f_add_parameter('xi_acquisition_function','RAND',comment='Set the acquisition strategy')
traj.f_add_parameter('m',30,comment='Number of pseudo-observations')



'''  ----- RUN EXPERIMENTS ------ '''

''' Start logging? '''
should_log = False
if should_log:
    orig_stdout = sys.stdout
    log_file = open('numerical_experiments_log_'+str(datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))+'.txt', "w")
    sys.stdout = log_file


''' Experiment settings: set acquisition strategies, etc. '''  
seeds = list(range(25))
#run_settings = {'initialization_seed':seeds, 
#                                  'xi_acquisition_function':['RAND']*len(seeds)}
run_settings = {'initialization_seed':seeds*5, 
                                  'xi_acquisition_function':['PCD','EXT','EI','EXR','RAND']*len(seeds)}
#run_settings = {'initialization_seed':seeds*7, 
#                                  'xi_acquisition_function':['PCD']*7*len(seeds),'m':[2,4,8,12,20,25,30]*len(seeds)}
traj.f_explore(run_settings)


''' Run experiment '''
start = time.time()
env.run(six_hump_camel)
#env.run(levy)
#env.run(hartmann6d)
#env.run(ackley)


### This is for debugging ###
#GP_model = env.run(six_hump_camel)
#X = GP_model[0][1].X
#Lambda = GP_model[0][1].Lambda_MAP
#mu_pred,Sigma_pred = GP_model[0][1].mu_Sigma_pred(X)
#Sigma = GP_model[0][1].Sigma
#Sigma_inv = GP_model[0][1].Sigma_inv
#posterior_covariance = GP_model[0][1].posterior_covariance
#posterior_covariance_inv = GP_model[0][1].posterior_covariance_inv


print("The session completed!")
print("Total time: " + str(time.time()-start) + " seconds.")

''' End logging '''
if should_log:
    sys.stdout = orig_stdout
    log_file.close()
''' ---------------'''



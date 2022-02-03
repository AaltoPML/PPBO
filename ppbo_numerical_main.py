#!/usr/bin/env python
#Petrus Mikkola (2020) @ Aalto University, Espoo, Finland

#To solve issues with openBLAS, MKL and Python multiprocessing (https://stackoverflow.com/questions/17053671/python-how-do-you-stop-numpy-from-multithreading)
#set the following environmental variables BEFORE opening Python
#export MKL_NUM_THREADS=1
#export OPENBLAS_NUM_THREADS=1
#export OMP_NUM_THREADS=1
#export NUMEXPR_NUM_THREADS=1

import os
import sys
#Set the correct working directory
wd_root = '/Users/mikkolp2/Desktop/PPBO/'
sys.path.insert(1, wd_root+'src/')
sys.path.insert(1, wd_root+'numerical_experiments/')
os.chdir(wd_root+'numerical_experiments/')

import pandas as pd
import numpy as np
import time
from datetime import datetime
from pypet import Environment

from ppbo_settings import PPBO_settings
from gp_model import GPModel
from acquisition import next_query
from misc import hypercube_corners
from test_functions import pp_ackley, pp_hartmann6d, pp_levy, pp_sixhump_camel, hartmann6d_orig, ackley_orig, levy_orig, sixhump_camel_orig


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

    mustar_results = [0]*(number_of_actual_queries+len(initial_queries_xi))
    xstar_results = np.empty([number_of_actual_queries+len(initial_queries_xi),PPBO_settings.D])
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
        if i==len(initial_queries_xi)-1:
            GP_model.turn_initialization_off()
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
        xstar = GP_model.FP.unscale(GP_model.xstar)
        mustar_results[i] = GP_model.mustar
        xstar_results[i,:] = xstar
        print("xstar of the initialization " + str(i+1)+"/"+str(len(initial_queries_xi))+ ' is ' + str(xstar))
        
    if OPTIMIZE_HYPERPARAMETERS_AFTER_INITIALIZATION:
        GP_model.update_model(optimize_theta=OPTIMIZE_HYPERPARAMETERS_AFTER_INITIALIZATION)  
    
    print('Initialization done! (Acq.' +str(PPBO_settings.xi_acquisition_function)+' )')
    GP_model.turn_initialization_off()
    
    ''' Queries -loop '''
    for i in range(number_of_actual_queries):
        print("Starting query " + str(i+1)+"/"+str(number_of_actual_queries)+" ...")
        if i+1==len(initial_queries_xi)+number_of_actual_queries:
            GP_model.set_last_iteration()
        ''' Compute next query '''
        xi_next,x_next = next_query(PPBO_settings,GP_model,unscale=True)
        ''' Query '''
        alpha_star = make_query(objective,xi_next,x_next)
        results = save_results(results,alpha_star*xi_next + x_next,xi_next,alpha_star)
        ''' Save results '''
        GP_model.update_feedback_processing_object(np.array(results))
        GP_model.mustar_previous_iteration = GP_model.mustar#GP_model.mustar()[1]
        ''' Update the model '''
        GP_model.update_data()
        if i+1==OPTIMIZE_HYPERPARAMETERS_AFTER_ACTUAL_QUERY_NUMBER:
            GP_model.update_model(optimize_theta=True)     
        else:
            GP_model.update_model(optimize_theta=OPTIMIZE_HYPERPARAMETERS_AFTER_EACH_ITERATION)
        xstar = GP_model.FP.unscale(GP_model.xstar)
        print("xstar of the iteration: " + str(xstar))
        print("The objective value at that point: " + str(-evaluate_objective(objective,xstar)))
        mustar_results[len(initial_queries_xi)+i] = GP_model.mustar
        xstar_results[len(initial_queries_xi)+i,:] = xstar

    print('Run done! (Acq.' +str(PPBO_settings.xi_acquisition_function)+' )')
    return results, xstar_results, mustar_results, GP_model



def six_hump_camel(traj):
    PPBO_settings_ = PPBO_settings(D=2,bounds=((-3,3),(-2,2)),
                                   xi_acquisition_function=traj.xi_acquisition_function,m=traj.m,
                                   theta_initial=[0.01,0.26,0.1],alpha_grid_distribution='equispaced') #[0.01,0.26,0.1],[1,0.1,8]
    np.random.seed(traj.initialization_seed) 
    initial_queries_xi = np.diag([PPBO_settings_.original_bounds[i][1] for i in range(PPBO_settings_.D)])
    #initial_queries_x = np.random.uniform([PPBO_settings_.original_bounds[i][0] for i in range(PPBO_settings_.D)], [PPBO_settings_.original_bounds[i][1] for i in range(PPBO_settings_.D)], (len(initial_queries_xi), PPBO_settings_.D))
    ''' Initial values = all cornes of the domain X '''
    initial_queries_xi = np.tile(initial_queries_xi,(2,1))
    initial_queries_x = hypercube_corners(PPBO_settings_.original_bounds)[0:len(initial_queries_xi)]
    results,xstar_results,mustar_results,GP_model = run_ppbo_loop('six_hump_camel',initial_queries_xi,initial_queries_x,traj.number_of_actual_queries,PPBO_settings_)
    traj.f_add_result('xstar',xstar_results)
    traj.f_add_result('mustar',mustar_results)
    return GP_model


def levy(traj):
    D = 10
    PPBO_settings_ = PPBO_settings(D=D,bounds=((-10,10),)*D,
                                   xi_acquisition_function=traj.xi_acquisition_function,m=traj.m,
                                   theta_initial=[0.001,0.4,0.15],alpha_grid_distribution='TGN') #[0.001,0.4,0.15],[1,0.1,8]
    initial_queries_xi = np.diag([PPBO_settings_.original_bounds[i][1] for i in range(PPBO_settings_.D)])
    np.random.seed(traj.initialization_seed) 
    initial_queries_x = np.random.uniform([PPBO_settings_.original_bounds[i][0] for i in range(PPBO_settings_.D)], [PPBO_settings_.original_bounds[i][1] for i in range(PPBO_settings_.D)], (len(initial_queries_xi), PPBO_settings_.D))
    results,xstar_results,mustar_results,GP_model= run_ppbo_loop('levy',initial_queries_xi,initial_queries_x,traj.number_of_actual_queries,PPBO_settings_)
    traj.f_add_result('xstar',xstar_results)
    traj.f_add_result('mustar',mustar_results)
    return GP_model

def ackley(traj):
    D = 20
    PPBO_settings_ = PPBO_settings(D=D,bounds=((-32.768, 32.768),)*D,
                                      xi_acquisition_function=traj.xi_acquisition_function,m=traj.m,
                                      theta_initial=[0.09,0.3,0.5],alpha_grid_distribution='TGN') #[0.09,0.3,0.5],[1,0.1,8]
    initial_queries_xi = np.diag([PPBO_settings_.original_bounds[i][1] for i in range(PPBO_settings_.D)])
    np.random.seed(traj.initialization_seed) 
    initial_queries_x = np.random.uniform([PPBO_settings_.original_bounds[i][0] for i in range(PPBO_settings_.D)], [PPBO_settings_.original_bounds[i][1] for i in range(PPBO_settings_.D)], (len(initial_queries_xi), PPBO_settings_.D))
    results,xstar_results,mustar_results,GP_model = run_ppbo_loop('ackley',initial_queries_xi,initial_queries_x,traj.number_of_actual_queries,PPBO_settings_)
    traj.f_add_result('xstar',xstar_results)
    traj.f_add_result('mustar',mustar_results)
    return GP_model

def hartmann6d(traj):
    PPBO_settings_ = PPBO_settings(D=6,bounds=((0, 1),)*6,
                                   xi_acquisition_function=traj.xi_acquisition_function,m=traj.m,
                                   theta_initial=[0.001,0.26,0.1],alpha_grid_distribution='TGN') #[0.001,0.26,0.1], [1,0.1,8]
    initial_queries_xi = np.eye(PPBO_settings_.D)
    np.random.seed(traj.initialization_seed) 
    initial_queries_x = np.random.uniform([PPBO_settings_.original_bounds[i][0] for i in range(PPBO_settings_.D)], [PPBO_settings_.original_bounds[i][1] for i in range(PPBO_settings_.D)], (len(initial_queries_xi), PPBO_settings_.D))
    results,xstar_results,mustar_results,GP_model = run_ppbo_loop('hartmann6d',initial_queries_xi,initial_queries_x,traj.number_of_actual_queries,PPBO_settings_)
    traj.f_add_result('xstar',xstar_results)
    traj.f_add_result('mustar',mustar_results)
    return GP_model

''' Run experimetns '''
NUMBER_OF_QUERIES = 35
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
traj.f_add_parameter('m',25,comment='Number of pseudo-observations')



'''  ----- RUN EXPERIMENTS ------ '''

''' Start logging? '''
should_log = False
if should_log:
    orig_stdout = sys.stdout
    log_file = open('numerical_experiments_log_'+str(datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))+'.txt', "w")
    sys.stdout = log_file


''' Experiment settings: set acquisition strategies, etc. '''  
seeds = list(range(1))
run_settings = {'initialization_seed':seeds, 
                                  'xi_acquisition_function':['PCD']*len(seeds)}
#run_settings = {'initialization_seed':seeds*10, 
#                                  'xi_acquisition_function':['PCD','EXT','RAND','EI','EXR']*len(seeds)}

traj.f_explore(run_settings)


''' Run experiment '''
start = time.time()
#env.run(six_hump_camel)
#env.run(levy)
env.run(hartmann6d)
#env.run(ackley)


print("The session completed!")
print("Total time: " + str(time.time()-start) + " seconds.")

''' End logging '''
if should_log:
    sys.stdout = orig_stdout
    log_file.close()
''' ---------------'''


























### THIS IS FOR DEBUGGING ###

# GP_model = env.run(six_hump_camel)
# GP_model = GP_model[0][1]
# #GP_model.theta = [1,0.09144542,10] #This is good
# #GP_model.update_model(optimize_theta=False)
# X = GP_model.X
# Lambda = GP_model.Lambda_MAP
# mu_pred,Sigma_pred = GP_model.mu_Sigma_pred(X)
# Sigma = pd.DataFrame(GP_model.Sigma)
# Sigma_inv = GP_model.Sigma_inv
# posterior_covariance = GP_model.posterior_covariance
# posterior_covariance_inv = GP_model.posterior_covariance_inv
# print(GP_model.FP.unscale(GP_model.xstar))
#
# from random_fourier_sampler import Hsampler
# h_sampler = Hsampler(GP_model)
# h_sampler.generate_basis()
# h_sampler.update_phi_X()
# h_sampler.update_omega_MAP()
# h_sampler.update_covariancematrix()
# cov_inv=pd.DataFrame(h_sampler.covariance_inv)
# cov=pd.DataFrame(h_sampler.covariance)
# xstar = h_sampler.return_xstar(h_sampler.omega_MAP)
# print(GP_model.FP.unscale(xstar))
#
# histogram=[]
# for i in range(200):
#     histogram.append(GP_model.FP.unscale(h_sampler.sample_xstar()))
# histogram = np.array(histogram)
# import matplotlib
# import matplotlib.pyplot as plt
# plt.plot(histogram[:,0],histogram[:,1],'ro',alpha=0.1)
# plt.scatter(0.0898,-0.7126, s=50,alpha=1,marker="*",color="black")
# plt.scatter(-0.0898,0.7126, s=50,alpha=1,marker="*",color="black")
# ############################ CONTOUR PLOTS ##########################
# import scipy
# import scipy.stats
# x = histogram[:, 0]
# y = histogram[:, 1]
# xmin = GP_model.original_bounds[0][0]
# xmax = GP_model.original_bounds[0][1]
# ymin = GP_model.original_bounds[1][0]
# ymax = GP_model.original_bounds[1][1]
# # Create meshgrid
# xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
# positions = np.vstack([xx.ravel(), yy.ravel()])
# values = np.vstack([x, y])
# kernel = scipy.stats.gaussian_kde(values)
# f = np.reshape(kernel(positions).T, xx.shape)
# fig = plt.figure(figsize=(8,8))
# ax = fig.gca()
# ax.set_xlim(xmin, xmax)
# ax.set_ylim(ymin, ymax)
# cfset = ax.contourf(xx, yy, f, cmap='coolwarm')
# ax.imshow(np.rot90(f), cmap='coolwarm', extent=[xmin, xmax, ymin, ymax])
# cset = ax.contour(xx, yy, f, colors='k')
# ax.clabel(cset, inline=1, fontsize=10)
# ax.scatter(0.0898,-0.7126, s=50,alpha=1,marker="*",color="black")
# ax.scatter(-0.0898,0.7126, s=50,alpha=1,marker="*",color="black")
# ax.set_xlabel("Variable $x_1$")
# ax.set_ylabel("Variable $x_2$")
# plt.title('2D Gaussian Kernel density estimation')
# #emprirical mean of the estimated distribution
# print('Mean: ' + str(np.mean(kernel.resample(100000),axis=1)))
# #################################################################################





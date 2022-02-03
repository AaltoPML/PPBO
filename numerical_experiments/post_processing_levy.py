import h5py
wd = '/u/18/mikkolp2/unix/Desktop/PPBO/'
import os
os.chdir(wd)

import numpy as np
import pandas as pd
from PPBO_numerical.test_functions import levy_orig, ackley_orig, dixonprice_orig, sixhump_camel_orig, hartmann6d_orig

#Plotting
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

SMALL_SIZE = 20
MEDIUM_SIZE = 25
BIGGER_SIZE = 28
matplotlib.rc('axes', titlesize=MEDIUM_SIZE)
matplotlib.rc('axes', labelsize=MEDIUM_SIZE)
matplotlib.rc('xtick', labelsize=SMALL_SIZE)    
matplotlib.rc('ytick', labelsize=SMALL_SIZE)    
matplotlib.rc('legend', fontsize=SMALL_SIZE)    

trajectory_name = 'numerical_experiments_trajectory_20-05-2020_20-43-36'

trajectory = h5py.File('hdf5/'+trajectory_name+'.hdf5', 'r')
TEST_FUNCTION_NAME = 'levy'

if TEST_FUNCTION_NAME == 'hartmann':
    D = 6
    TRUE_GLOBAL_MINIMIZER = np.array([0.20169,0.150011,0.476874,0.275332,0.311652,0.6573])
if TEST_FUNCTION_NAME == 'ackley':
    D = 20
    TRUE_GLOBAL_MINIMIZER = np.array([0]*D)
if TEST_FUNCTION_NAME == 'levy':
    D = 10
    TRUE_GLOBAL_MINIMIZER = np.array([1]*D)


def function_value(x):
    if TEST_FUNCTION_NAME == 'hartmann':
        return -hartmann6d_orig(x)
    elif TEST_FUNCTION_NAME == 'sixhump_camel':
        return -sixhump_camel_orig(x)
    elif TEST_FUNCTION_NAME == 'ackley':
        return -ackley_orig(x)
    elif TEST_FUNCTION_NAME == 'levy':
        return -levy_orig(x)
    else:
        return None
def distance_to_global_minimizer(x):
   if TEST_FUNCTION_NAME == 'sixhump_camel':
       true_global_minimizer1 = np.array([0.0898,-0.7126])
       true_global_minimizer2 = np.array([-0.0898,0.7126])
       RMSE1 = np.sqrt(sum(np.square(x-true_global_minimizer1)))
       RMSE2 = np.sqrt(sum(np.square(x-true_global_minimizer2)))
       if RMSE1 < RMSE2:
           return RMSE1
       else:
           return RMSE2
       
   return np.sqrt(sum(np.square(x-TRUE_GLOBAL_MINIMIZER)))



parameters = trajectory[trajectory_name]['parameters']
acquisition_strategies = parameters['xi_acquisition_function']['explored_data'][:].astype(str)


results = trajectory[trajectory_name]['results']
runs = results['runs']

n_initial_queries = 10
n_actual_queries = 90
n_seeds = 25 #but what if there are not completed runs?
completed_runs_n = {a : 0 for a in acquisition_strategies}

results_f_value = pd.DataFrame(data=np.zeros((n_initial_queries+n_actual_queries,len(set(acquisition_strategies)))),columns=sorted(list(set(acquisition_strategies))), dtype=np.float64)
results_x_dist = pd.DataFrame(data=np.zeros((n_initial_queries+n_actual_queries,len(set(acquisition_strategies)))),columns=sorted(list(set(acquisition_strategies))), dtype=np.float64)
ei = np.empty(shape=[0, n_initial_queries+n_actual_queries])
ext = np.empty(shape=[0, n_initial_queries+n_actual_queries])
exr = np.empty(shape=[0, n_initial_queries+n_actual_queries])
pcd = np.empty(shape=[0, n_initial_queries+n_actual_queries])
rand = np.empty(shape=[0, n_initial_queries+n_actual_queries])
ei_xmin = np.empty(shape=[0, n_initial_queries+n_actual_queries])
ext_xmin = np.empty(shape=[0, n_initial_queries+n_actual_queries])
exr_xmin = np.empty(shape=[0, n_initial_queries+n_actual_queries])
pcd_xmin = np.empty(shape=[0, n_initial_queries+n_actual_queries])
rand_xmin = np.empty(shape=[0, n_initial_queries+n_actual_queries])

i = 0
for run in list(runs.keys()):
    a_startegy = acquisition_strategies[i]
    data = runs[run]
    data = data['x_star']
    data = data.get('x_star').value
    print(a_startegy)
    completed_runs_n[a_startegy] = completed_runs_n[a_startegy] + 1
    print([function_value(data[t,:]) for t in range(len(data))])
    if all(results_f_value.loc[:,a_startegy] == np.zeros((n_initial_queries+n_actual_queries,))):
        results_f_value.loc[:,a_startegy] = [function_value(data[t,:]) for t in range(len(data))]
    else:
        results_f_value.loc[:,a_startegy] = results_f_value.loc[:,a_startegy] + [function_value(data[t,:]) for t in range(len(data))]
    if all(results_x_dist.loc[:,a_startegy] == np.zeros((n_initial_queries+n_actual_queries,))):
        results_x_dist.loc[:,a_startegy] = [distance_to_global_minimizer(data[t,:]) for t in range(len(data))]
    else:
        results_x_dist.loc[:,a_startegy] = results_x_dist.loc[:,a_startegy] + [distance_to_global_minimizer(data[t,:]) for t in range(len(data))]   
    
    if a_startegy=='EI':
        ei = np.vstack([ei,[function_value(data[t,:]) for t in range(len(data))]])
        ei_xmin = np.vstack([ei_xmin,[distance_to_global_minimizer(data[t,:]) for t in range(len(data))]])
    if a_startegy=='EXT':   
        ext = np.vstack([ext,[function_value(data[t,:]) for t in range(len(data))]])
        ext_xmin = np.vstack([ext_xmin,[distance_to_global_minimizer(data[t,:]) for t in range(len(data))]])
    if a_startegy=='EXR':
        exr = np.vstack([exr,[function_value(data[t,:]) for t in range(len(data))]])
        exr_xmin = np.vstack([exr_xmin,[distance_to_global_minimizer(data[t,:]) for t in range(len(data))]])
    if a_startegy=='PCD':
        pcd = np.vstack([pcd,[function_value(data[t,:]) for t in range(len(data))]])
        pcd_xmin = np.vstack([pcd_xmin,[distance_to_global_minimizer(data[t,:]) for t in range(len(data))]])
    if a_startegy=='RAND':
        rand = np.vstack([rand,[function_value(data[t,:]) for t in range(len(data))]])
        rand_xmin = np.vstack([rand_xmin,[distance_to_global_minimizer(data[t,:]) for t in range(len(data))]])
    
    
    i = i + 1


results_f_value = results_f_value/n_seeds
results_x_dist = results_x_dist/n_seeds
#for a in results_f_value.columns:
#    results_f_value.loc[:,a] = results_f_value.loc[:,a]/completed_runs_n[a]
#    results_x_dist.loc[:,a] = results_x_dist.loc[:,a]/completed_runs_n[a]
ei = np.std(ei,axis=0)
ext = np.std(ext,axis=0) 
exr = np.std(exr,axis=0) 
pcd = np.std(pcd,axis=0) 
rand = np.std(rand,axis=0)    
ei_xmin = np.std(ei_xmin,axis=0)
ext_xmin = np.std(ext_xmin,axis=0) 
exr_xmin = np.std(exr_xmin,axis=0) 
pcd_xmin = np.std(pcd_xmin,axis=0) 
rand_xmin = np.std(rand_xmin,axis=0)    











''' PBO results '''

PBO_trajectory_name = 'PBO_numerical_experiments_trajectory_28-05-2020_08-41-19'

PBO_trajectory = h5py.File('hdf5/'+PBO_trajectory_name+'.hdf5', 'r')
PBO_results = PBO_trajectory[PBO_trajectory_name]['results']
objectives = PBO_trajectory[PBO_trajectory_name]['parameters']
objectives = objectives['objective']['explored_data'][:].astype(str)
PBO_runs = PBO_results['runs']
for run in list(PBO_runs.keys()):
    print(run)
    data = PBO_runs[run]
    data = data['mean_y']
    data = data.get('mean_y').value

pbo_dts = PBO_runs[list(PBO_runs.keys())[0]]   
pbo_dts_mean = pbo_dts['mean_y'].get('mean_y').value
pbo_dts_sd = pbo_dts['y_res'].get('y_res').value 
pbo_dts_sd = np.std(pbo_dts_sd,axis=0)
pbo_dts_xmin = pbo_dts['x_res'].get('x_res').value
pbo_dts_xmin = np.array([distance_to_global_minimizer(x) for x in pbo_dts_xmin]).reshape(100,100)
pbo_dts_xmin_mean = np.mean(pbo_dts_xmin,axis=0)
pbo_dts_xmin_sd = np.std(pbo_dts_xmin,axis=0)


pbo = PBO_runs[list(PBO_runs.keys())[1]]
pbo_mean = pbo['mean_y'].get('mean_y').value 
pbo_sd = pbo['y_res'].get('y_res').value
pbo_sd = np.std(pbo_sd,axis=0)
pbo_xmin = pbo['x_res'].get('x_res').value
pbo_xmin = np.array([distance_to_global_minimizer(x) for x in pbo_xmin]).reshape(100,100)
pbo_xmin_mean = np.mean(pbo_xmin,axis=0)
pbo_xmin_sd = np.std(pbo_xmin,axis=0)











''' ---- plotting -------------------------------'''


acquisition_strategies = sorted(list(set(acquisition_strategies)))
print(acquisition_strategies)
x_axis_points = list(range(1,n_initial_queries+n_actual_queries+1))
linestyles=['solid','dotted','dashed','dashdot','solid']
cmap = plt.get_cmap('ocean')
colors = cmap(np.linspace(0, 1.0, len(acquisition_strategies)+2)) #plus 2 for PBO
colors[len(acquisition_strategies)+1] = [0.5,0.5,0.5,1]
linewidth=3.7
elinewidth=0.8
errobar_lowlimit = np.array([-1.0316]*len(x_axis_points))


''' -------------- function value ----------------------'''
plt.figure(0)
''' Plot PPBO'''
for strategy,color,linestyle in zip(acquisition_strategies,colors[0:len(acquisition_strategies)],linestyles):
    y_axis_points = results_f_value.loc[:,strategy].values 
    if strategy=='EI':
        plt.errorbar(x_axis_points,y_axis_points, yerr=ei, color=color, linestyle=linestyle,linewidth=linewidth, elinewidth=elinewidth)
    if strategy=='EXT':
        plt.errorbar(x_axis_points,y_axis_points, yerr=ext, color=color, linestyle=linestyle,linewidth=linewidth, elinewidth=elinewidth)
    if strategy=='EXR':
        plt.errorbar(x_axis_points,y_axis_points, yerr=exr, color=color, linestyle=linestyle,linewidth=linewidth, elinewidth=elinewidth)
    if strategy=='PCD':
        plt.errorbar(x_axis_points,y_axis_points, yerr=pcd, color=color, linestyle=linestyle,linewidth=linewidth, elinewidth=elinewidth)
    if strategy=='RAND':
        plt.errorbar(x_axis_points,y_axis_points, yerr=rand, color=color, linestyle=linestyle,linewidth=linewidth, elinewidth=elinewidth)
''' Plot PBO'''
plt.errorbar(x_axis_points,pbo_mean,marker='.', yerr=pbo_sd, color=colors[len(acquisition_strategies)],linewidth=linewidth, elinewidth=elinewidth)
plt.errorbar(x_axis_points,pbo_dts_mean,marker='o', yerr=pbo_dts_sd, color=colors[len(acquisition_strategies)+1],linewidth=linewidth, elinewidth=elinewidth)
''' PPlot settings '''
legend = ['PPBO-'+str(aq) for aq in acquisition_strategies] + ['PBO-RAND','PBO-DTS']
#plt.legend(legend)
plt.xlabel("Number of iterations")
plt.ylabel("$f(\mathbf{x}_{opt})$")
plt.title('Levy10D', fontsize=BIGGER_SIZE)
plt.hlines(0, 0, n_initial_queries+n_actual_queries+1, colors='k', linestyles='dashed')
plt.ylim(-1, 135)
plt.show()
 


'''------------------- distance to global minimizer ------------------------'''
plt.figure(1)
''' Plot PPBO'''
for strategy,color,linestyle in zip(acquisition_strategies,colors[0:len(acquisition_strategies)],linestyles):
    y_axis_points = results_x_dist.loc[:,strategy].values 
    if strategy=='EI':
        plt.errorbar(x_axis_points,y_axis_points, yerr=ei_xmin, color=color, linestyle=linestyle,linewidth=linewidth, elinewidth=elinewidth)
    if strategy=='EXT':
        plt.errorbar(x_axis_points,y_axis_points, yerr=ext_xmin, color=color, linestyle=linestyle,linewidth=linewidth, elinewidth=elinewidth)
    if strategy=='EXR':
        plt.errorbar(x_axis_points,y_axis_points, yerr=exr_xmin, color=color, linestyle=linestyle,linewidth=linewidth, elinewidth=elinewidth)
    if strategy=='PCD':
        plt.errorbar(x_axis_points,y_axis_points, yerr=pcd_xmin, color=color, linestyle=linestyle,linewidth=linewidth, elinewidth=elinewidth)
    if strategy=='RAND':
        plt.errorbar(x_axis_points,y_axis_points, yerr=rand_xmin, color=color, linestyle=linestyle,linewidth=linewidth, elinewidth=elinewidth)
''' Plot PBO'''
plt.errorbar(x_axis_points,pbo_xmin_mean,marker='.', yerr=pbo_xmin_sd, color=colors[len(acquisition_strategies)],linewidth=linewidth, elinewidth=elinewidth)
plt.errorbar(x_axis_points,pbo_dts_xmin_mean,marker='o', yerr=pbo_dts_xmin_sd, color=colors[len(acquisition_strategies)+1],linewidth=linewidth, elinewidth=elinewidth)
''' PPlot settings '''
legend = ['PPBO-'+str(aq) for aq in acquisition_strategies] + ['PBO-RAND','PBO-DTS']
#plt.legend(legend)
plt.xlabel("Number of iterations")
plt.ylabel("Distance to the global minimizer")
plt.title('Levy10D', fontsize=BIGGER_SIZE)
plt.hlines(0, 0, n_initial_queries+n_actual_queries+1, colors='k', linestyles='dashed')
plt.ylim(-0.2, 20)
plt.show()
















    
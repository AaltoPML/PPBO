import h5py
wd = '/u/18/mikkolp2/unix/Desktop/PPBO/'
#wd = '/u/18/mikkolp2/unix/Documents/ackley20d'
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


trajectory_name = 'numerical_experiments_trajectory_20-05-2020_20-41-34'

trajectory = h5py.File('hdf5/'+trajectory_name+'.hdf5', 'r')
TEST_FUNCTION_NAME = 'hartmann'

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

n_initial_queries = 6
n_actual_queries = 94
n_seeds = 25 #but what if there are not completed runs?
completed_runs_n = {a : 0 for a in acquisition_strategies}

results_f_value = pd.DataFrame(data=np.zeros((n_initial_queries+n_actual_queries,len(set(acquisition_strategies)))),columns=set(acquisition_strategies), dtype=np.float64)
results_x_dist = pd.DataFrame(data=np.zeros((n_initial_queries+n_actual_queries,len(set(acquisition_strategies)))),columns=set(acquisition_strategies), dtype=np.float64)
i = 0
for run in list(runs.keys()):
    #d_acquisition = d_acquisitions[i]
    #x_acquisition = x_acquisitions[i]
    #a_startegy = d_acquisition+'-'+x_acquisition
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
    i = i + 1



#results_f_value = results_f_value/n_seeds
#results_x_dist = results_x_dist/n_seeds
for a in results_f_value.columns:
    results_f_value.loc[:,a] = results_f_value.loc[:,a]/completed_runs_n[a]
    results_x_dist.loc[:,a] = results_x_dist.loc[:,a]/completed_runs_n[a]
    















''' PBO results '''

PBO_trajectory_name = 'PBO_numerical_experiments_trajectory_29-04-2020_16-15-12'
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

   

pbo_sixhump_dts = PBO_runs[list(PBO_runs.keys())[0]]
pbo_sixhump_dts = pbo_sixhump_dts['mean_y']
pbo_sixhump_dts = pbo_sixhump_dts.get('mean_y').value 
pbo_sixhump_dts[0:5] = [14,]*5
pbo_sixhump = PBO_runs[list(PBO_runs.keys())[1]]
pbo_sixhump = pbo_sixhump['mean_y']
pbo_sixhump = pbo_sixhump.get('mean_y').value 
pbo_sixhump[0:5] = [14,]*5


pbo_hartmann_dts = PBO_runs[list(PBO_runs.keys())[2]]
pbo_hartmann_dts_mean = pbo_hartmann_dts['mean_y']
pbo_hartmann_dts_mean = pbo_hartmann_dts_mean.get('mean_y').value 
pbo_hartmann_dts_sd = pbo_hartmann_dts['y_res']
pbo_hartmann_dts_sd = pbo_hartmann_dts_sd.get('y_res').value  
pbo_hartmann_dts_sd = np.std(pbo_hartmann_dts_sd,axis=0)
pbo_hartmann = PBO_runs[list(PBO_runs.keys())[3]]
pbo_hartmann_mean = pbo_hartmann['mean_y']
pbo_hartmann_mean = pbo_hartmann_mean.get('mean_y').value 
pbo_hartmann_sd = pbo_hartmann['y_res']
pbo_hartmann_sd = pbo_hartmann_sd.get('y_res').value  
pbo_hartmann_sd = np.std(pbo_hartmann_sd,axis=0)




#Error during runs
#pbo_levy_dts = PBO_runs[list(PBO_runs.keys())[0]]
#pbo_levy_dts = pbo_levy_dts['mean_y']
#pbo_levy_dts = pbo_levy_dts.get('mean_y').value 
#pbo_levy_dts[0:5] = [14,]*5
pbo_levy = PBO_runs[list(PBO_runs.keys())[4]]
pbo_levy = pbo_levy['mean_y']
pbo_levy = pbo_levy.get('mean_y').value 
pbo_levy[0:5] = [140,]*5

pbo_ackley_dts = PBO_runs[list(PBO_runs.keys())[5]]
pbo_ackley_dts = pbo_ackley_dts['mean_y']
pbo_ackley_dts = pbo_ackley_dts.get('mean_y').value 
pbo_ackley_dts[0:5] = [22,]*5
pbo_ackley = PBO_runs[list(PBO_runs.keys())[6]]
pbo_ackley = pbo_ackley['mean_y']
pbo_ackley = pbo_ackley.get('mean_y').value 
pbo_ackley[0:5] = [22,]*5


















''' ---- plotting -------------------------------'''


acquisition_strategies = list(set(acquisition_strategies))
print(acquisition_strategies)
x_axis_points = list(range(1,n_initial_queries+n_actual_queries+1))
linestyles=['solid','dotted','dashed','dashdot','solid']
cmap = plt.get_cmap('jet')
colors = cmap(np.linspace(0, 1.0, len(acquisition_strategies)+2)) #plus 2 for PBO
linewidth=3


''' -------------- function value ----------------------'''
''' Plot PPBO'''
for strategy,color,linestyle in zip(acquisition_strategies,colors[0:len(acquisition_strategies)],linestyles):
    y_axis_points = results_f_value.loc[:,strategy].values 
    plt.plot(x_axis_points,y_axis_points, color=color, linestyle=linestyle,linewidth=linewidth)
''' Plot PBO'''
#plt.plot(x_axis_points,pbo_sixhump,color=colors[len(acquisition_strategies)],linewidth=linewidth)
plt.errorbar(x_axis_points,pbo_hartmann_mean,marker='o', yerr=pbo_hartmann_sd, color=colors[len(acquisition_strategies)],linewidth=linewidth, elinewidth=0.5)
#plt.plot(x_axis_points,pbo_sixhump_dts, ,color=colors[len(acquisition_strategies)+1],linewidth=linewidth)
plt.errorbar(x_axis_points,pbo_hartmann_dts_mean,marker='o', yerr=pbo_hartmann_dts_sd, color=colors[len(acquisition_strategies)+1],linewidth=linewidth, elinewidth=0.5)
''' PPlot settings '''
legend = ['PPBO-'+str(aq) for aq in acquisition_strategies] + ['PBO-RAND','PBO-DTS']
plt.legend(legend)
plt.xlabel("Number of iterations")
plt.ylabel("$f(\mathbf{x}_{opt})$")
plt.title('Hartmann6D', fontsize=BIGGER_SIZE)
plt.hlines(-3.322, 0, n_initial_queries+n_actual_queries+1, colors='k', linestyles='dashed')
plt.ylim(-3.5, 0.5)
plt.show()
 
'''------------------- distance to global minimizer ------------------------'''
''' Plot PPBO'''
for strategy,color,linestyle in zip(acquisition_strategies,colors[0:len(acquisition_strategies)],linestyles):
    y_axis_points = results_x_dist.loc[:,strategy].values 
    plt.plot(x_axis_points,y_axis_points, color=color, linestyle=linestyle)
''' Plot PBO'''
plt.plot(x_axis_points,[distance_to_global_minimizer(x) for x in pbo_hartmann_mean],color=colors[len(acquisition_strategies)])
plt.plot(x_axis_points,[distance_to_global_minimizer(x) for x in pbo_hartmann_dts_mean], marker='o',color=colors[len(acquisition_strategies)+1])
''' PPlot settings '''
legend = ['PPBO-'+str(aq) for aq in acquisition_strategies] + ['PBO-RAND','PBO-DTS']
plt.legend(legend)
plt.xlabel("Number of iterations")
plt.ylabel("Distance to the global minimizer")
plt.title('Hartmann6D', fontsize=BIGGER_SIZE)
plt.hlines(0, 0, n_initial_queries+n_actual_queries+1, colors='k', linestyles='dashed')
plt.ylim(-0.2, 3)
plt.show()












    
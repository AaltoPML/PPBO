import numpy as np
import scipy
import scipy.stats
import time

GP_model = GP_model[0][1]

#Plotting
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

''' Plotting '''
slice_1_dim = 1
slice_2_dim = 2



''' f_MAP '''
fig = plt.figure()
ax = plt.axes(projection="3d")
ax.view_init(elev=90, azim=-90)
ax.scatter3D(GP_model.X[:,slice_1_dim-1], GP_model.X[:,slice_2_dim-1], GP_model.f_MAP, c=GP_model.f_MAP, cmap='hsv');
plt.show()

''' Posterior Mode location'''
x_star_, mu_star_  = GP_model.mu_star()
x_star_ = GP_model.FP.unscale(x_star_)
print(x_star_)





###################### Argmin-distribution ###############################
D = 2
var_bounds = GP_model.bounds

''' Create grid '''
grid_x1, grid_x2  = np.mgrid[var_bounds[0][0]:var_bounds[0][1]:40j,
                                                                 var_bounds[1][0]:var_bounds[1][1]:40j]

def create_grid(*args):
    for i in range(0,D-1):
        grid1 = args[i]
        grid2 = args[i+1]
        if i == 0:
            grid = np.vstack([grid1.ravel(), grid2.ravel()])
        else:
            grid = np.vstack([grid,grid2.ravel()])
    return(grid.T)     
grid=create_grid(grid_x1, grid_x2)
del grid_x1, grid_x2




''' Slices plots of the posterior mean'''
slice_1_dim = 1
slice_2_dim = 2
''' f_MAP '''
fig = plt.figure()
ax = plt.axes(projection="3d")
ax.view_init(elev=90, azim=-90)
f_postmean = GP_model.mu_Sigma_pred(grid)[0]
unscaled_grid = GP_model.FP.unscale(grid)
ax.scatter3D(unscaled_grid[:,slice_1_dim-1], unscaled_grid[:,slice_2_dim-1], f_postmean, c=f_postmean, cmap='hsv');
ax.scatter3D(0.0898,-0.7126,np.max(f_postmean)*1.05, s=50,alpha=1,marker="*",color="black") #Sixhump-test function global minimizer
ax.scatter3D(-0.0898,0.7126,np.max(f_postmean)*1.05, s=50,alpha=1,marker="*",color="black") #Sixhump-test function global minimizer
plt.show()
print(unscaled_grid[:,[slice_1_dim-1,slice_2_dim-1]][np.where(f_postmean == np.max(f_postmean))[0],:])
''' ------------------------------------------'''







''' Generate Argmin-distribution '''

#############################################
def generate_argmin_posterior(GP_model,sample_size):
    n_input_points = 500
    sample = np.empty((sample_size,D))
    for i in range(0,sample_size):
        print(i)
        input_points = np.random.random((n_input_points, D)) #Unif([0.0, 1.0))
        unscaled_input_points = GP_model.FP.unscale(input_points)
        mu_,Sigma_= GP_model.mu_Sigma_pred(input_points)
        #print(mu_)
        #print(np.diag(Sigma_))
        f_values = list(np.random.multivariate_normal(mu_,Sigma_)) #predict/sample GP
        argmax = unscaled_input_points[np.where(f_values == np.max(f_values))[0],:] #or unsclae grid before?
        sample[i] = list(argmax[0])
    return(sample)
start = time.time()
histogram = generate_argmin_posterior(GP_model,50)
print(time.time()-start)
#####################################
#def generate_argmin_posterior(post_mean,post_covariance,sample_size):
#    sample = np.empty((sample_size,D))
#    unscaled_grid = GP_model.FP.unscale(grid)
#    for i in range(0,sample_size):
#        f_values = list(np.random.multivariate_normal(post_mean,post_covariance)) #predict/sample GP
#        argmax = unscaled_grid[np.where(f_values == np.max(f_values))[0],:] #or unsclae grid before?
#        sample[i] = list(argmax[0])
#    return(sample)
#f_post_mean,f_post_covar = GP_model.mu_Sigma_pred(grid)  
#start = time.time()
#histogram = generate_argmin_posterior(f_post_mean,f_post_covar,100)
#print(time.time()-start)



############## Plotting ##################################
mean_ = [np.mean(histogram[:,slice_1_dim-1]),np.mean(histogram[:,slice_2_dim-1])]
mode_ = [scipy.stats.mode(histogram[:,slice_1_dim-1])[0],scipy.stats.mode(histogram[:,slice_2_dim-1])[0]]
plt.plot(histogram[:,slice_1_dim-1],histogram[:,slice_2_dim-1],'ro',alpha=0.1)
plt.axis([GP_model.original_bounds[0][0],GP_model.original_bounds[0][1],GP_model.original_bounds[0][0],GP_model.original_bounds[0][1]])
plt.scatter(mean_[0],mean_[1], s=50,alpha=1,marker="x",color="blue")
plt.scatter(mode_[0],mode_[1], s=50,alpha=1,marker="x",color="purple")
plt.scatter(0.0898,-0.7126, s=50,alpha=1,marker="*",color="black")
plt.scatter(-0.0898,0.7126, s=50,alpha=1,marker="*",color="black")
plt.legend(["posterior draw","posterior mean","posterior mode","true global minimizer 1","true global minimizer 1"])
plt.xlabel("Variable $x_1$")
plt.ylabel("Variable $x_2$")
plt.show()
################################################################



############################ CONTOUR PLOTS ###########################
x = histogram[:, 0]
y = histogram[:, 1]
xmin = GP_model.original_bounds[0][0]
xmax = GP_model.original_bounds[0][1]
ymin = GP_model.original_bounds[1][0]
ymax = GP_model.original_bounds[1][1]
# Create meshgrid
xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
positions = np.vstack([xx.ravel(), yy.ravel()])
values = np.vstack([x, y])
kernel = scipy.stats.gaussian_kde(values)
f = np.reshape(kernel(positions).T, xx.shape)
fig = plt.figure(figsize=(8,8))
ax = fig.gca()
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
cfset = ax.contourf(xx, yy, f, cmap='coolwarm')
ax.imshow(np.rot90(f), cmap='coolwarm', extent=[xmin, xmax, ymin, ymax])
cset = ax.contour(xx, yy, f, colors='k')
ax.clabel(cset, inline=1, fontsize=10)
ax.scatter(0.0898,-0.7126, s=50,alpha=1,marker="*",color="black")
ax.scatter(-0.0898,0.7126, s=50,alpha=1,marker="*",color="black")
ax.set_xlabel("Variable $x_1$")
ax.set_ylabel("Variable $x_2$")
plt.title('2D Gaussian Kernel density estimation')
#emprirical mean of the estimated distribution
print('Mean: ' + str(np.mean(kernel.resample(100000),axis=1)))
#################################################################################




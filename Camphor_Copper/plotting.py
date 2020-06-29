import scipy
import scipy.stats

#Plotting
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

from misc import unscale
from datetime import datetime

''' f_MAP '''
slice_1_dim = 4
slice_2_dim = 5
fig = plt.figure()
ax = plt.axes(projection="3d")
ax.view_init(elev=29, azim=--53)
ax.scatter3D(GP_model.X[:,slice_1_dim-1], GP_model.X[:,slice_2_dim-1], GP_model.f_MAP, c=GP_model.f_MAP, cmap='hsv');
ax.set_xlabel("\u03B1")
ax.set_ylabel("\u03B2")
plt.title('$\mathbf{f}_{MAP}$')
plt.show()
slice_1_dim = 1
slice_2_dim = 2
fig = plt.figure()
ax = plt.axes(projection="3d")
ax.view_init(elev=29, azim=-53)
ax.scatter3D(GP_model.X[:,slice_1_dim-1], GP_model.X[:,slice_2_dim-1], GP_model.f_MAP, c=GP_model.f_MAP, cmap='hsv');
ax.set_xlabel("x")
ax.set_ylabel("y")
plt.title('$\mathbf{f}_{MAP}$')
plt.show()
slice_1_dim = 5
slice_2_dim = 6
fig = plt.figure()
ax = plt.axes(projection="3d")
ax.view_init(elev=29, azim=-53)
ax.scatter3D(GP_model.X[:,slice_1_dim-1], GP_model.X[:,slice_2_dim-1], GP_model.f_MAP, c=GP_model.f_MAP, cmap='hsv');
ax.set_xlabel("\u03B2")
ax.set_ylabel("\u03B3")
plt.title('$\mathbf{f}_{MAP}$')
plt.show()


''' Posterior Mode Configuration '''
x_star_, mu_star_  = GP_model.mu_star(mu_star_finding_trials=50)
x_star_unscaled = GP_model.FP.unscale_conf(x_star_)
dict_x_star = dict(zip(['camp_dx','camp_dy','camp_origin_height','alpha','beta','gamma'],x_star_unscaled))
print(dict_x_star)
#Visualize the optimal configuration
import ase
import ase.visualize  
from create_111_camphor_func import create_file
create_file(**dict_x_star) #x_star
system = ase.io.read('geometry.in')
HTML = ase.visualize.view(system, viewer="x3d").data   
file = open('optimal_x_'+str(datetime.now().strftime("%d-%m-%Y_%H-%M-%S"))+'.html','w') 
file.write(HTML) 
file.close()









###################### Argmin-distribution,etc ALPHA + BETA ###############################
D = 6
var_bounds = GP_model.bounds
''' Create grid '''
grid_x1, grid_x2, grid_x3, grid_x4, grid_x5, grid_x6  = np.mgrid[x_star_[0]:x_star_[0]:1j,
                                                                 x_star_[0]:x_star_[0]:1j,
                                                                 x_star_[0]:x_star_[0]:1j,
                                                                 var_bounds[3][0]:var_bounds[3][1]:33j,
                                                                 var_bounds[4][0]:var_bounds[4][1]:33j,
                                                                 x_star_[5]:x_star_[5]:1j]
def create_grid(*args):
    for i in range(0,D-1):
        grid1 = args[i]
        grid2 = args[i+1]
        if i == 0:
            grid = np.vstack([grid1.ravel(), grid2.ravel()])
        else:
            grid = np.vstack([grid,grid2.ravel()])
    return(grid.T)     
grid=create_grid(grid_x1, grid_x2, grid_x3, grid_x4, grid_x5, grid_x6)
del grid_x1, grid_x2, grid_x3, grid_x4, grid_x5, grid_x6



''' Slices plots of the posterior mean'''
slice_1_dim = 4
slice_2_dim = 5
fig = plt.figure()
ax = plt.axes(projection="3d")
ax.view_init(elev=90, azim=-90)
f_postmean = GP_model.mu_Sigma_pred(grid)[0]
unscaled_grid = unscale(grid,GP_model.FP.X_nonscaled)
ax.scatter3D(unscaled_grid[:,slice_1_dim-1], unscaled_grid[:,slice_2_dim-1], f_postmean, c=f_postmean, cmap='hsv');
ax.set_xlabel("\u03B1")
ax.set_ylabel("\u03B2")
plt.title('\u03BC($\mathbf{x}$)')
plt.show()
print(unscaled_grid[:,[slice_1_dim-1,slice_2_dim-1]][np.where(f_postmean == np.max(f_postmean))[0],:])
''' ------------------------------------------'''


''' Generate Argmin-distribution '''
#Argmin plots not so good when few queries since unexplored areas warp the results 
def generate_argmin_posterior(post_mean,post_covariance,sample_size):
    sample = np.empty((sample_size,D))
    unscaled_grid = unscale(grid,GP_model.FP.X_nonscaled)
    for i in range(0,sample_size):
        f_values = list(np.random.multivariate_normal(post_mean,post_covariance)) #predict/sample GP
        argmax = unscaled_grid[np.where(f_values == np.max(f_values))[0],:] #or unsclae grid before?
        sample[i] = list(argmax[0])
    return(sample)

f_post_mean,f_post_covar = GP_model.mu_Sigma_pred(grid)        

start = time.time()
histogram = generate_argmin_posterior(f_post_mean,f_post_covar,1000)
print(time.time()-start)



xmin = -180
xmax = 180
ymin = -180
ymax = 180
############## Histogram ##################################
mean_ = [np.mean(histogram[:,3]),np.mean(histogram[:,4])]
mode_ = [float(scipy.stats.mode(histogram[:,3])[0]),float(scipy.stats.mode(histogram[:,4])[0])]
plt.plot(histogram[:,3],histogram[:,4],'ro',alpha=0.1)
plt.axis([xmin,xmax,ymin,ymax])
plt.scatter(mode_[0],mode_[1], s=50,alpha=1,marker="x",color="blue")
plt.scatter(x_star_unscaled[3],x_star_unscaled[4], s=50,alpha=1,marker="*",color="black")
plt.legend(["posterior draw","posterior mode","posterior mean minimizer"])
plt.xlabel("\u03B1")
plt.ylabel("\u03B2")
plt.show()
################################################################
############################ CONTOUR PLOTS ###########################
x = histogram[:, 3]
y = histogram[:, 4]
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
#ax.scatter(0.0898,-0.7126, s=50,alpha=1,marker="*",color="black")
#ax.scatter(-0.0898,0.7126, s=50,alpha=1,marker="*",color="black")
ax.set_xlabel("\u03B1")
ax.set_ylabel("\u03B2")
plt.title('Gaussian Kernel density estimation of argmax distribution')
##########################################################################################################











































###################### Argmin-distribution,etc x + y ###############################
D = 6
var_bounds = GP_model.bounds

''' Create grid '''
grid_x1, grid_x2, grid_x3, grid_x4, grid_x5, grid_x6  = np.mgrid[var_bounds[0][0]:var_bounds[0][1]:33j,
                                                                 var_bounds[1][0]:var_bounds[1][1]:33j,
                                                                 x_star_[2]:x_star_[2]:1j,
                                                                 x_star_[3]:x_star_[3]:1j,
                                                                 x_star_[4]:x_star_[4]:1j,
                                                                 x_star_[5]:x_star_[5]:1j]
def create_grid(*args):
    for i in range(0,D-1):
        grid1 = args[i]
        grid2 = args[i+1]
        if i == 0:
            grid = np.vstack([grid1.ravel(), grid2.ravel()])
        else:
            grid = np.vstack([grid,grid2.ravel()])
    return(grid.T)     
grid=create_grid(grid_x1, grid_x2, grid_x3, grid_x4, grid_x5, grid_x6)
del grid_x1, grid_x2, grid_x3, grid_x4, grid_x5, grid_x6



''' Slices plots of the posterior mean'''
slice_1_dim = 1
slice_2_dim = 2
fig = plt.figure()
ax = plt.axes(projection="3d")
ax.view_init(elev=90, azim=-90)
f_postmean = GP_model.mu_Sigma_pred(grid)[0]
unscaled_grid = unscale(grid,GP_model.FP.X_nonscaled)
ax.scatter3D(unscaled_grid[:,slice_1_dim-1], unscaled_grid[:,slice_2_dim-1], f_postmean, c=f_postmean, cmap='hsv');
ax.set_xlabel("x")
ax.set_ylabel("y")
plt.title('\u03BC($\mathbf{x}$)')
plt.show()
print(unscaled_grid[:,[slice_1_dim-1,slice_2_dim-1]][np.where(f_postmean == np.max(f_postmean))[0],:])
''' ------------------------------------------'''


''' Generate Argmin-distribution '''
#Argmin plots not so good when few queries since unexplored areas warp the results 
def generate_argmin_posterior(post_mean,post_covariance,sample_size):
    sample = np.empty((sample_size,D))
    unscaled_grid = unscale(grid,GP_model.FP.X_nonscaled)
    for i in range(0,sample_size):
        f_values = list(np.random.multivariate_normal(post_mean,post_covariance)) #predict/sample GP
        argmax = unscaled_grid[np.where(f_values == np.max(f_values))[0],:] #or unsclae grid before?
        sample[i] = list(argmax[0])
    return(sample)

f_post_mean,f_post_covar = GP_model.mu_Sigma_pred(grid)        

start = time.time()
histogram = generate_argmin_posterior(f_post_mean,f_post_covar,1000)
print(time.time()-start)



xmin = -0.5
xmax = 0.5
ymin = -0.5
ymax = 0.5
############## Histogram ##################################
mean_ = [np.mean(histogram[:,0]),np.mean(histogram[:,1])]
mode_ = [float(scipy.stats.mode(histogram[:,0])[0]),float(scipy.stats.mode(histogram[:,1])[0])]
plt.plot(histogram[:,0],histogram[:,1],'ro',alpha=0.1)
plt.axis([xmin,xmax,ymin,ymax])
plt.scatter(mode_[0],mode_[1], s=50,alpha=1,marker="x",color="blue")
plt.scatter(x_star_unscaled[0],x_star_unscaled[1], s=50,alpha=1,marker="*",color="black")
plt.legend(["posterior draw","posterior mode","posterior mean minimizer"])
plt.xlabel("x")
plt.ylabel("y")
plt.show()
################################################################
############################ CONTOUR PLOTS ###########################
x = histogram[:, 0]
y = histogram[:, 1]
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
#ax.scatter(0.0898,-0.7126, s=50,alpha=1,marker="*",color="black")
#ax.scatter(-0.0898,0.7126, s=50,alpha=1,marker="*",color="black")
ax.set_xlabel("x")
ax.set_ylabel("y")
plt.title('Gaussian Kernel density estimation of argmax distribution')
##########################################################################################################












###################### Argmin-distribution,etc y + z ###############################
D = 6
var_bounds = GP_model.bounds

''' Create grid '''
grid_x1, grid_x2, grid_x3, grid_x4, grid_x5, grid_x6  = np.mgrid[x_star_[0]:x_star_[0]:1j,
                                                                 var_bounds[1][0]:var_bounds[1][1]:33j,
                                                                 var_bounds[2][0]:var_bounds[2][1]:33j,
                                                                 x_star_[3]:x_star_[3]:1j,
                                                                 x_star_[4]:x_star_[4]:1j,
                                                                 x_star_[5]:x_star_[5]:1j]
def create_grid(*args):
    for i in range(0,D-1):
        grid1 = args[i]
        grid2 = args[i+1]
        if i == 0:
            grid = np.vstack([grid1.ravel(), grid2.ravel()])
        else:
            grid = np.vstack([grid,grid2.ravel()])
    return(grid.T)     
grid=create_grid(grid_x1, grid_x2, grid_x3, grid_x4, grid_x5, grid_x6)
del grid_x1, grid_x2, grid_x3, grid_x4, grid_x5, grid_x6



''' Slices plots of the posterior mean'''
slice_1_dim = 2
slice_2_dim = 3
fig = plt.figure()
ax = plt.axes(projection="3d")
ax.view_init(elev=90, azim=-90)
f_postmean = GP_model.mu_Sigma_pred(grid)[0]
unscaled_grid = unscale(grid,GP_model.FP.X_nonscaled)
ax.scatter3D(unscaled_grid[:,slice_1_dim-1], unscaled_grid[:,slice_2_dim-1], f_postmean, c=f_postmean, cmap='hsv');
ax.set_xlabel("y")
ax.set_ylabel("z")
plt.title('\u03BC($\mathbf{x}$)')
plt.show()
print(unscaled_grid[:,[slice_1_dim-1,slice_2_dim-1]][np.where(f_postmean == np.max(f_postmean))[0],:])
''' ------------------------------------------'''


''' Generate Argmin-distribution '''
#Argmin plots not so good when few queries since unexplored areas warp the results 
def generate_argmin_posterior(post_mean,post_covariance,sample_size):
    sample = np.empty((sample_size,D))
    unscaled_grid = unscale(grid,GP_model.FP.X_nonscaled)
    for i in range(0,sample_size):
        f_values = list(np.random.multivariate_normal(post_mean,post_covariance)) #predict/sample GP
        argmax = unscaled_grid[np.where(f_values == np.max(f_values))[0],:] #or unsclae grid before?
        sample[i] = list(argmax[0])
    return(sample)

f_post_mean,f_post_covar = GP_model.mu_Sigma_pred(grid)        

start = time.time()
histogram = generate_argmin_posterior(f_post_mean,f_post_covar,1000)
print(time.time()-start)



xmin = -0.5
xmax = 0.5
ymin = 4
ymax = 7
############## Histogram ##################################
mean_ = [np.mean(histogram[:,slice_1_dim-1]),np.mean(histogram[:,slice_2_dim-1])]
mode_ = [float(scipy.stats.mode(histogram[:,slice_1_dim-1])[0]),float(scipy.stats.mode(histogram[:,slice_2_dim-1])[0])]
plt.plot(histogram[:,slice_1_dim-1],histogram[:,slice_2_dim-1],'ro',alpha=0.1)
plt.axis([xmin,xmax,ymin,ymax])
plt.scatter(mode_[0],mode_[1], s=50,alpha=1,marker="x",color="blue")
plt.scatter(x_star_unscaled[slice_1_dim-1],x_star_unscaled[slice_2_dim-1], s=50,alpha=1,marker="*",color="black")
plt.legend(["posterior draw","posterior mode","posterior mean minimizer"])
plt.xlabel("y")
plt.ylabel("z")
plt.show()
################################################################
############################ CONTOUR PLOTS ###########################
x = histogram[:, slice_1_dim-1]
y = histogram[:, slice_2_dim-1]
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
#ax.scatter(0.0898,-0.7126, s=50,alpha=1,marker="*",color="black")
#ax.scatter(-0.0898,0.7126, s=50,alpha=1,marker="*",color="black")
ax.set_xlabel("y")
ax.set_ylabel("z")
plt.title('Gaussian Kernel density estimation of argmax distribution')
##########################################################################################################


























###################### Argmin-distribution,etc beta + gamma ###############################
D = 6
var_bounds = GP_model.bounds

''' Create grid '''
grid_x1, grid_x2, grid_x3, grid_x4, grid_x5, grid_x6  = np.mgrid[x_star_[0]:x_star_[0]:1j,
                                                                 x_star_[1]:x_star_[1]:1j,
                                                                 x_star_[2]:x_star_[2]:1j,
                                                                 x_star_[3]:x_star_[3]:1j,
                                                                 var_bounds[4][0]:var_bounds[4][1]:33j,
                                                                 var_bounds[5][0]:var_bounds[5][1]:33j]
def create_grid(*args):
    for i in range(0,D-1):
        grid1 = args[i]
        grid2 = args[i+1]
        if i == 0:
            grid = np.vstack([grid1.ravel(), grid2.ravel()])
        else:
            grid = np.vstack([grid,grid2.ravel()])
    return(grid.T)     
grid=create_grid(grid_x1, grid_x2, grid_x3, grid_x4, grid_x5, grid_x6)
del grid_x1, grid_x2, grid_x3, grid_x4, grid_x5, grid_x6



''' Slices plots of the posterior mean'''
slice_1_dim = 5
slice_2_dim = 6
fig = plt.figure()
ax = plt.axes(projection="3d")
ax.view_init(elev=90, azim=-90)
f_postmean = GP_model.mu_Sigma_pred(grid)[0]
unscaled_grid = unscale(grid,GP_model.FP.X_nonscaled)
ax.scatter3D(unscaled_grid[:,slice_1_dim-1], unscaled_grid[:,slice_2_dim-1], f_postmean, c=f_postmean, cmap='hsv');
ax.set_xlabel("\u03B2")
ax.set_ylabel("\u03B3")
plt.title('\u03BC($\mathbf{x}$)')
plt.show()
print(unscaled_grid[:,[slice_1_dim-1,slice_2_dim-1]][np.where(f_postmean == np.max(f_postmean))[0],:])
''' ------------------------------------------'''


''' Generate Argmin-distribution '''
#Argmin plots not so good when few queries since unexplored areas warp the results 
def generate_argmin_posterior(post_mean,post_covariance,sample_size):
    sample = np.empty((sample_size,D))
    unscaled_grid = unscale(grid,GP_model.FP.X_nonscaled)
    for i in range(0,sample_size):
        f_values = list(np.random.multivariate_normal(post_mean,post_covariance)) #predict/sample GP
        argmax = unscaled_grid[np.where(f_values == np.max(f_values))[0],:] #or unsclae grid before?
        sample[i] = list(argmax[0])
    return(sample)

f_post_mean,f_post_covar = GP_model.mu_Sigma_pred(grid)        

start = time.time()
histogram = generate_argmin_posterior(f_post_mean,f_post_covar,1000)
print(time.time()-start)



xmin = -180
xmax = 180
ymin = -180
ymax = 180
############## Histogram ##################################
mean_ = [np.mean(histogram[:,slice_1_dim-1]),np.mean(histogram[:,slice_2_dim-1])]
mode_ = [float(scipy.stats.mode(histogram[:,slice_1_dim-1])[0]),float(scipy.stats.mode(histogram[:,slice_2_dim-1])[0])]
plt.plot(histogram[:,slice_1_dim-1],histogram[:,slice_2_dim-1],'ro',alpha=0.1)
plt.axis([xmin,xmax,ymin,ymax])
plt.scatter(mode_[0],mode_[1], s=50,alpha=1,marker="x",color="blue")
plt.scatter(x_star_unscaled[slice_1_dim-1],x_star_unscaled[slice_2_dim-1], s=50,alpha=1,marker="*",color="black")
plt.legend(["posterior draw","posterior mode","posterior mean minimizer"])
plt.xlabel("\u03B2")
plt.ylabel("\u03B3")
plt.show()
################################################################
############################ CONTOUR PLOTS ###########################
x = histogram[:, slice_1_dim-1]
y = histogram[:, slice_2_dim-1]
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
#ax.scatter(0.0898,-0.7126, s=50,alpha=1,marker="*",color="black")
#ax.scatter(-0.0898,0.7126, s=50,alpha=1,marker="*",color="black")
ax.set_xlabel("\u03B2")
ax.set_ylabel("\u03B3")
plt.title('Gaussian Kernel density estimation of argmax distribution')
##########################################################################################################


















###################### Argmin-distribution,etc alpha + gamma ###############################
D = 6
var_bounds = GP_model.bounds

''' Create grid '''
grid_x1, grid_x2, grid_x3, grid_x4, grid_x5, grid_x6  = np.mgrid[x_star_[0]:x_star_[0]:1j,
                                                                 x_star_[1]:x_star_[1]:1j,
                                                                 x_star_[2]:x_star_[2]:1j,
                                                                 var_bounds[3][0]:var_bounds[3][1]:33j,
                                                                 x_star_[4]:x_star_[4]:1j,
                                                                 var_bounds[5][0]:var_bounds[5][1]:33j]
def create_grid(*args):
    for i in range(0,D-1):
        grid1 = args[i]
        grid2 = args[i+1]
        if i == 0:
            grid = np.vstack([grid1.ravel(), grid2.ravel()])
        else:
            grid = np.vstack([grid,grid2.ravel()])
    return(grid.T)     
grid=create_grid(grid_x1, grid_x2, grid_x3, grid_x4, grid_x5, grid_x6)
del grid_x1, grid_x2, grid_x3, grid_x4, grid_x5, grid_x6



''' Slices plots of the posterior mean'''
slice_1_dim = 4
slice_2_dim = 6
fig = plt.figure()
ax = plt.axes(projection="3d")
ax.view_init(elev=90, azim=-90)
f_postmean = GP_model.mu_Sigma_pred(grid)[0]
unscaled_grid = unscale(grid,GP_model.FP.X_nonscaled)
ax.scatter3D(unscaled_grid[:,slice_1_dim-1], unscaled_grid[:,slice_2_dim-1], f_postmean, c=f_postmean, cmap='hsv');
ax.set_xlabel("\u03B1")
ax.set_ylabel("\u03B3")
plt.title('\u03BC($\mathbf{x}$)')
plt.show()
print(unscaled_grid[:,[slice_1_dim-1,slice_2_dim-1]][np.where(f_postmean == np.max(f_postmean))[0],:])
''' ------------------------------------------'''


''' Generate Argmin-distribution '''
#Argmin plots not so good when few queries since unexplored areas warp the results 
def generate_argmin_posterior(post_mean,post_covariance,sample_size):
    sample = np.empty((sample_size,D))
    unscaled_grid = unscale(grid,GP_model.FP.X_nonscaled)
    for i in range(0,sample_size):
        f_values = list(np.random.multivariate_normal(post_mean,post_covariance)) #predict/sample GP
        argmax = unscaled_grid[np.where(f_values == np.max(f_values))[0],:] #or unsclae grid before?
        sample[i] = list(argmax[0])
    return(sample)

f_post_mean,f_post_covar = GP_model.mu_Sigma_pred(grid)        

start = time.time()
histogram = generate_argmin_posterior(f_post_mean,f_post_covar,1000)
print(time.time()-start)



xmin = -180
xmax = 180
ymin = -180
ymax = 180
############## Histogram ##################################
mean_ = [np.mean(histogram[:,slice_1_dim-1]),np.mean(histogram[:,slice_2_dim-1])]
mode_ = [float(scipy.stats.mode(histogram[:,slice_1_dim-1])[0]),float(scipy.stats.mode(histogram[:,slice_2_dim-1])[0])]
plt.plot(histogram[:,slice_1_dim-1],histogram[:,slice_2_dim-1],'ro',alpha=0.1)
plt.axis([xmin,xmax,ymin,ymax])
plt.scatter(mode_[0],mode_[1], s=50,alpha=1,marker="x",color="blue")
plt.scatter(x_star_unscaled[slice_1_dim-1],x_star_unscaled[slice_2_dim-1], s=50,alpha=1,marker="*",color="black")
plt.legend(["posterior draw","posterior mode","posterior mean minimizer"])
plt.xlabel("\u03B2")
plt.ylabel("\u03B3")
plt.show()
################################################################
############################ CONTOUR PLOTS ###########################
x = histogram[:, slice_1_dim-1]
y = histogram[:, slice_2_dim-1]
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
#ax.scatter(0.0898,-0.7126, s=50,alpha=1,marker="*",color="black")
#ax.scatter(-0.0898,0.7126, s=50,alpha=1,marker="*",color="black")
ax.set_xlabel("\u03B2")
ax.set_ylabel("\u03B3")
plt.title('Gaussian Kernel density estimation of argmax distribution')
##########################################################################################################
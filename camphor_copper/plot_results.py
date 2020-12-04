import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d



def sliceplot_pred_mean(var1,var2,GP_model,x_star_):
    D = 6
    var_bounds = GP_model.bounds
    def create_grid(*args):
        for i in range(0,D-1):
            grid1 = args[i]
            grid2 = args[i+1]
            if i == 0:
                grid = np.vstack([grid1.ravel(), grid2.ravel()])
            else:
                grid = np.vstack([grid,grid2.ravel()])
        return(grid.T) 
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.view_init(elev=90, azim=-90)
    ''' ------- Different variants var1,var2 in Combinations(x,y,z,alpha,beta,gamma) ---------'''
    if var1=='alpha' and var2=='beta':
        ''' Create grid '''
        grid_x1, grid_x2, grid_x3, grid_x4, grid_x5, grid_x6  = np.mgrid[x_star_[0]:x_star_[0]:1j,
                                                                         x_star_[0]:x_star_[0]:1j,
                                                                         x_star_[0]:x_star_[0]:1j,
                                                                         var_bounds[3][0]:var_bounds[3][1]:33j,
                                                                         var_bounds[4][0]:var_bounds[4][1]:33j,
                                                                         x_star_[5]:x_star_[5]:1j]    
        grid=create_grid(grid_x1, grid_x2, grid_x3, grid_x4, grid_x5, grid_x6)
        ''' Slices plots of the posterior mean'''
        slice_1_dim = 4
        slice_2_dim = 5
        f_postmean = GP_model.mu_Sigma_pred(grid)[0]
        unscaled_grid = GP_model.FP.unscale(grid)
        ax.scatter3D(unscaled_grid[:,slice_1_dim-1], unscaled_grid[:,slice_2_dim-1], f_postmean, c=f_postmean, cmap='hsv');
        ax.set_xlabel("\u03B1")
        ax.set_ylabel("\u03B2")
        #print(unscaled_grid[:,[slice_1_dim-1,slice_2_dim-1]][np.where(f_postmean == np.max(f_postmean))[0],:])
    if var1=='x' and var2=='y':
        ''' Create grid '''
        grid_x1, grid_x2, grid_x3, grid_x4, grid_x5, grid_x6  = np.mgrid[var_bounds[0][0]:var_bounds[0][1]:33j,
                                                                         var_bounds[1][0]:var_bounds[1][1]:33j,
                                                                         x_star_[2]:x_star_[2]:1j,
                                                                         x_star_[3]:x_star_[3]:1j,
                                                                         x_star_[4]:x_star_[4]:1j,
                                                                         x_star_[5]:x_star_[5]:1j]
        grid=create_grid(grid_x1, grid_x2, grid_x3, grid_x4, grid_x5, grid_x6)
        ''' Slices plots of the posterior mean'''
        slice_1_dim = 1
        slice_2_dim = 2
        f_postmean = GP_model.mu_Sigma_pred(grid)[0]
        unscaled_grid = GP_model.FP.unscale(grid)
        ax.scatter3D(unscaled_grid[:,slice_1_dim-1], unscaled_grid[:,slice_2_dim-1], f_postmean, c=f_postmean, cmap='hsv');
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    if var1=='z' and var2=='gamma':
        ''' Create grid '''
        grid_x1, grid_x2, grid_x3, grid_x4, grid_x5, grid_x6  = np.mgrid[x_star_[0]:x_star_[0]:1j,
                                                                         x_star_[1]:x_star_[1]:1j,
                                                                         var_bounds[2][0]:var_bounds[2][1]:33j,
                                                                         x_star_[3]:x_star_[3]:1j,
                                                                         x_star_[4]:x_star_[4]:1j,
                                                                         var_bounds[5][0]:var_bounds[5][1]:33j]
        grid=create_grid(grid_x1, grid_x2, grid_x3, grid_x4, grid_x5, grid_x6)
        ''' Slices plots of the posterior mean'''
        slice_1_dim = 3
        slice_2_dim = 6
        f_postmean = GP_model.mu_Sigma_pred(grid)[0]
        unscaled_grid = GP_model.FP.unscale(grid)
        ax.scatter3D(unscaled_grid[:,slice_1_dim-1], unscaled_grid[:,slice_2_dim-1], f_postmean, c=f_postmean, cmap='hsv');
        ax.set_xlabel("z")
        ax.set_ylabel("\u03B3")
    ''' ------------------------------------------------------ ''' 
    plt.title('\u03BC($\mathbf{x}$)')
    plt.show()

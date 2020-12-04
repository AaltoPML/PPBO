import numpy as np

def dist(X1,X2):
    """
    Compute the Euclidean distance between each row of X1 and X2
    """
    X1sq = np.sum(np.square(X1),1)  
    X2sq = np.sum(np.square(X2),1)
    sqdist = -2.*np.dot(X1, X2.T) + (X1sq[:,None] + X2sq[None,:])
    sqdist = np.clip(sqdist, 0, np.inf)
    return sqdist 


def d(x1,x2):
    ''' Compute absolute distance between all elments of two vectors and output outersubtract matrix ''' 
    return np.abs(np.subtract.outer(x1,x2))


def SE_kernel(X1,X2,theta):
    l = theta[1]
    sigma_f = theta[2]
    if l <=0 or sigma_f <= 0:
        print("Check hyperparameter values!")
    sqdist = dist(X1,X2)
    return sigma_f**2 * np.exp(-0.5*sqdist/(l**2))  
        
def RQ_kernel(X1,X2,theta):
    alpha = 2
    l = theta[1]
    sigma_f = theta[2]
    if l <=0 or sigma_f <= 0:
        print("Check hyperparameter values!")
    sqdist = dist(X1,X2)
    return sigma_f**2 * (1+sqdist/(2*alpha*l**2))**(-alpha)

def camphor_copper_kernel(X1,X2,theta):
    #This hyperparameter vector will work well:  theta=[0.001,0.26,0.1] (given that 0.05 added to lengthscale of RBF for z-variable)
    #OLD hyperparameters that stood out in testing: theta=[0.09,0.2,0.35]
    l = theta[1]
    sigma_f = theta[2]
    if l <=0 or sigma_f <= 0:
        print("Check hyperparameter values!")
    ''' Same lenghtscale l across dimensions '''
    ''' Periodic kernels has same period p=1 since data normalized to [0,1] '''
    p = 1
    kernelX = np.exp((-2*np.square(np.sin(np.pi*d(X1[:,0],X2[:,0])/p)))/l**2)
    kernelY = np.exp((-2*np.square(np.sin(np.pi*d(X1[:,1],X2[:,1])/p)))/l**2)
    kernelZ = np.exp(-0.5*np.square(d(X1[:,2],X2[:,2]))/((l+0.05)**2)) 
    ''' NOTE: 0.05 added to lengthscale of RBF for z-variable'''
    kernelalpha = np.exp((-2*np.square(np.sin(np.pi*d(X1[:,3],X2[:,3])/p)))/l**2)
    kernelbeta = np.exp((-2*np.square(np.sin(np.pi*d(X1[:,4],X2[:,4])/p)))/l**2)
    kernelgamma = np.exp((-2*np.square(np.sin(np.pi*d(X1[:,5],X2[:,5])/p)))/l**2)
    return sigma_f**2 * kernelX * kernelY * kernelZ * kernelalpha * kernelbeta * kernelgamma 
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


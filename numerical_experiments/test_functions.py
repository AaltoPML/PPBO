from __future__ import division
import numpy as np
from numpy import abs, cos, exp, mean, pi, prod, sin, sqrt, sum
import scipy
import scipy.stats
from misc import alpha_bounds



''' Find alpha_star Ackley'''
def pp_ackley(xi,x):
    lower = [-32.768]*len(xi)
    upper = [32.768]*len(xi)
    #Alpha bounds!!
    lower,upper = alpha_bounds(xi,lower,upper)
    bounds = list(zip([lower],[upper]))
    res = scipy.optimize.differential_evolution(ackley, bounds, args=(xi,x),updating='immediate', disp=False, tol=0.001) 
    return float(res.x)


def ackley(alpha,xi,x):
    return -ackley_orig(alpha*xi + x)

#x = np.array([0,0.15,0.47687,0.27533,0.31165,0.6573])
#xi = np.array([1,0,0,0,0,0])
#x = np.array([0,0,0,0,0,0.6573])
#xi = np.array([1,0.2,0,1,0.1,0])
def pp_hartmann6d(xi,x):
    lower = [0]*len(xi)
    upper = [1]*len(xi)
    #Alpha bounds!!
    lower,upper = alpha_bounds(xi,lower,upper)
    bounds = list(zip([lower],[upper]))
    res = scipy.optimize.differential_evolution(hartmann6d, bounds, args=(xi,x),updating='immediate', disp=False, tol=0.001) 
    return float(res.x)

def hartmann6d(alpha,xi,x):
    return -hartmann6d_orig(alpha*xi + x)


def pp_levy(xi,x):
    lower = [-10]*len(xi)
    upper = [10]*len(xi)
    #Alpha bounds!!
    lower,upper = alpha_bounds(xi,lower,upper)
    bounds = list(zip([lower],[upper]))
    res = scipy.optimize.differential_evolution(levy, bounds, args=(xi,x),updating='immediate', disp=False, tol=0.001) 
    return float(res.x)

def levy(alpha,xi,x):
    return -levy_orig(alpha*xi + x)


def pp_sixhump_camel(xi,x):
    lower = [-3]*len(xi)
    upper = [3]*len(xi)
    #Alpha bounds!!
    lower,upper = alpha_bounds(xi,lower,upper)
    bounds = list(zip([lower],[upper]))
    res = scipy.optimize.differential_evolution(sixhump_camel, bounds, args=(xi,x),updating='immediate', disp=False, tol=0.001) 
    return float(res.x)

def sixhump_camel(alpha,xi,x):
    return -sixhump_camel_orig(alpha*xi + x)


#Give CORRECT x_minus_d and d: This determines the dimension of the input fro functions!


#

''' Find argmin Levy'''
def argmin_levy(x_minus_d,d):
    lower = [-10]
    upper = [10]
    bounds = list(zip(lower,upper))
    res = scipy.optimize.differential_evolution(levy, bounds, args=(x_minus_d,d),updating='immediate', disp=False) 
    return float(res.x)


''' Find argmin Ackley'''
def argmin_ackley(x_minus_d,d):
    lower = [-32.768]
    upper = [32.768]
    bounds = list(zip(lower,upper))
    res = scipy.optimize.differential_evolution(ackley, bounds, args=(x_minus_d,d),updating='immediate', disp=False) 
    return float(res.x)


''' Find argmin dixonprice'''
def argmin_dixonprice(x_minus_d,d):
    lower = [-10]
    upper = [10]
    bounds = list(zip(lower,upper))
    res = scipy.optimize.differential_evolution(dixonprice, bounds, args=(x_minus_d,d),updating='immediate', disp=False) 
    return float(res.x)


''' Find argmin dixonprice'''
def argmin_sixhump_camel(x_minus_d,d):
    if d==1:
        bounds = list(zip([-3],[3]))
    else:
        bounds = list(zip([-2],[2]))
    res = scipy.optimize.differential_evolution(sixhump_camel, bounds, args=(x_minus_d,d),updating='immediate', disp=False) 
    return float(res.x)

''' Find argmin hartman6d'''
def argmin_hartmann6d(x_minus_d,d):
    lower = [0]
    upper = [1]
    bounds = list(zip(lower,upper))
    res = scipy.optimize.differential_evolution(hartmann6d, bounds, args=(x_minus_d,d),updating='immediate', disp=False)
    return float(res.x)





''' ALLWAYS NEGATIVE OF TRUE TEST FUNCTIONS IS CONSIDERED! (orig definition) '''
''' AND NEGATIVE OF ARGUMENT FUNCTIONS TO GET ARGMAX! '''


'''' NOISE LEVEL!! '''
NOISE_LEVEL = 0.001 #0.001 



#...............................................................................
def levy_orig( x ):
    x = np.asarray_chkfinite(x)
    #print(len(x)) # is same as D
    z = 1 + (x - 1) / 4
    
    return -((sin( pi * z[0] )**2
        + sum( (z[:-1] - 1)**2 * (1 + 10 * sin( pi * z[:-1] + 1 )**2 ))
        +       (z[-1] - 1)**2 * (1 + sin( 2 * pi * z[-1] )**2 ))) + + np.abs(60-0)*np.random.normal(0,NOISE_LEVEL)
#...............................................................................

#def levy(x_d,x_minus_d,d):
#    x_ = list(x_minus_d)
#    x_.insert(d-1,float(x_d))
#    return -levy_orig(x_)
    
    
def ackley_orig( x, a=20, b=0.2, c=2*pi ):
    x = np.asarray_chkfinite(x)  # ValueError if any NaN or Inf
    n = len(x)
    s1 = sum( x**2 )
    s2 = sum( cos( c * x ))
    return -(-a*exp( -b*sqrt( s1 / n )) - exp( s2 / n ) + a + exp(1)) + np.random.normal(0,NOISE_LEVEL) #np.abs(21-0)*np.random.normal(0,NOISE_LEVEL)

#def ackley(x_d,x_minus_d,d):
#    x_ = list(x_minus_d)
#    x_.insert(d-1,float(x_d))
#    return -ackley_orig(x_)


def dixonprice_orig( x ):  # dp.m
    x = np.asarray_chkfinite(x)
    n = len(x)
    j = np.arange( 2, n+1 )
    x2 = 2 * x**2
    return -(sum( j * (x2[1:] - x[:-1]) **2 ) + (x[0] - 1) **2) + np.random.normal(0,NOISE_LEVEL)

#def dixonprice(x_d,x_minus_d,d):
#    x_ = list(x_minus_d)
#    x_.insert(d-1,float(x_d))
#    return -dixonprice_orig(x_)



#Six-Hump Camel: x in [-3,3] and y in [-2,2]
def sixhump_camel_orig(x):
    x = np.asarray_chkfinite(x)
    return -float((4 - 2.1*(x[0]**2) + (x[0]**4)/3)*(x[0]**2) + x[0]*x[1] + (-4 + 4*(x[1]**2))*(x[1]**2)) +  np.random.normal(0,NOISE_LEVEL)   #  np.abs(100-(-1.0316))*np.random.normal(0,NOISE_LEVEL)

#def sixhump_camel(x_d,x_minus_d,d):
#    x_ = list(x_minus_d)
#    x_.insert(d-1,float(x_d))
#    return -sixhump_camel_orig(x_)


#Hartman 6D
def hartmann6d_orig(xx):  
  alpha = np.array([1.0, 1.2, 3.0, 3.2]).T
  A = np.array([[10, 3, 17, 3.5, 1.7, 8],
         [0.05, 10, 17, 0.1, 8, 14],
         [3, 3.5, 1.7, 10, 17, 8],
         [17, 8, 0.05, 10, 0.1, 14]])
  P = (10**(-4))*np.array([[1312, 1696, 5569, 124, 8283, 5886],
                   [2329, 4135, 8307, 3736, 1004, 9991],
                   [2348, 1451, 3522, 2883, 3047, 6650],
                   [4047, 8828, 8732, 5743, 1091, 381]])
  xxmat = np.array([list(xx),list(xx),list(xx),list(xx)]) #xxmat <- matrix(rep(xx,times=4), 4, 6, byrow=TRUE)
  inner = np.sum(A*np.power(xxmat-P,2),axis=1)
  outer = sum(alpha*np.exp(-inner))
  y = -outer
  return -y + np.random.normal(0,NOISE_LEVEL)  #np.abs(0-(-3.32237))*np.random.normal(0,NOISE_LEVEL)

#def hartmann6d(x_d,x_minus_d,d):
#    x_ = list(x_minus_d)
#    x_.insert(d-1,float(x_d))
#    return -hartmann6d_orig(x_)





#
#
#
#
##...............................................................................
#def ackley( x, a=20, b=0.2, c=2*pi ):
#    x = np.asarray_chkfinite(x)  # ValueError if any NaN or Inf
#    n = len(x)
#    s1 = sum( x**2 )
#    s2 = sum( cos( c * x ))
#    return -a*exp( -b*sqrt( s1 / n )) - exp( s2 / n ) + a + exp(1)
#
##...............................................................................
#def dixonprice( x ):  # dp.m
#    x = np.asarray_chkfinite(x)
#    n = len(x)
#    j = np.arange( 2, n+1 )
#    x2 = 2 * x**2
#    return sum( j * (x2[1:] - x[:-1]) **2 ) + (x[0] - 1) **2
#
##...............................................................................
#def griewank( x, fr=4000 ):
#    x = np.asarray_chkfinite(x)
#    n = len(x)
#    j = np.arange( 1., n+1 )
#    s = sum( x**2 )
#    p = prod( cos( x / sqrt(j) ))
#    return s/fr - p + 1
#
###...............................................................................
##def levy( x ):
##    x = np.asarray_chkfinite(x)
##    n = len(x)
##    z = 1 + (x - 1) / 4
##    return (sin( pi * z[0] )**2
##        + sum( (z[:-1] - 1)**2 * (1 + 10 * sin( pi * z[:-1] + 1 )**2 ))
##        +       (z[-1] - 1)**2 * (1 + sin( 2 * pi * z[-1] )**2 ))
##
###...............................................................................
#michalewicz_m = .5  # orig 10: ^20 => underflow
#
#def michalewicz( x ):  # mich.m
#    x = np.asarray_chkfinite(x)
#    n = len(x)
#    j = np.arange( 1., n+1 )
#    return - sum( sin(x) * sin( j * x**2 / pi ) ** (2 * michalewicz_m) )
#
##...............................................................................
#def perm( x, b=.5 ):
#    x = np.asarray_chkfinite(x)
#    n = len(x)
#    j = np.arange( 1., n+1 )
#    xbyj = np.fabs(x) / j
#    return mean([ mean( (j**k + b) * (xbyj ** k - 1) ) **2
#            for k in j/n ])
#    # original overflows at n=100 --
#    # return sum([ sum( (j**k + b) * ((x / j) ** k - 1) ) **2
#    #       for k in j ])
#
##...............................................................................
#def powell( x ):
#    x = np.asarray_chkfinite(x)
#    n = len(x)
#    n4 = ((n + 3) // 4) * 4
#    if n < n4:
#        x = np.append( x, np.zeros( n4 - n ))
#    x = x.reshape(( 4, -1 ))  # 4 rows: x[4i-3] [4i-2] [4i-1] [4i]
#    f = np.empty_like( x )
#    f[0] = x[0] + 10 * x[1]
#    f[1] = sqrt(5) * (x[2] - x[3])
#    f[2] = (x[1] - 2 * x[2]) **2
#    f[3] = sqrt(10) * (x[0] - x[3]) **2
#    return sum( f**2 )
#
##...............................................................................
#def powersum( x, b=[8,18,44,114] ):  # power.m
#    x = np.asarray_chkfinite(x)
#    n = len(x)
#    s = 0
#    for k in range( 1, n+1 ):
#        bk = b[ min( k - 1, len(b) - 1 )]  # ?
#        s += (sum( x**k ) - bk) **2  # dim 10 huge, 100 overflows
#    return s
#
##...............................................................................
#def rastrigin( x ):  # rast.m
#    x = np.asarray_chkfinite(x)
#    n = len(x)
#    return 10*n + sum( x**2 - 10 * cos( 2 * pi * x ))
#
##...............................................................................
#def rosenbrock( x ):  # rosen.m
#    """ http://en.wikipedia.org/wiki/Rosenbrock_function """
#        # a sum of squares, so LevMar (scipy.optimize.leastsq) is pretty good
#    x = np.asarray_chkfinite(x)
#    x0 = x[:-1]
#    x1 = x[1:]
#    return (sum( (1 - x0) **2 )
#        + 100 * sum( (x1 - x0**2) **2 ))
#
##...............................................................................
#def schwefel( x ):  # schw.m
#    x = np.asarray_chkfinite(x)
#    n = len(x)
#    return 418.9829*n - sum( x * sin( sqrt( abs( x ))))
#
##...............................................................................
#def sphere( x ):
#    x = np.asarray_chkfinite(x)
#    return sum( x**2 )
#
##...............................................................................
#def sum2( x ):
#    x = np.asarray_chkfinite(x)
#    n = len(x)
#    j = np.arange( 1., n+1 )
#    return sum( j * x**2 )
#
##...............................................................................
#def trid( x ):
#    x = np.asarray_chkfinite(x)
#    return sum( (x - 1) **2 ) - sum( x[:-1] * x[1:] )
#
##...............................................................................
#def zakharov( x ):  # zakh.m
#    x = np.asarray_chkfinite(x)
#    n = len(x)
#    j = np.arange( 1., n+1 )
#    s2 = sum( j * x ) / 2
#    return sum( x**2 ) + s2**2 + s2**4
#
##...............................................................................
#    # not in Hedar --
#
#def ellipse( x ):
#    x = np.asarray_chkfinite(x)
#    return mean( (1 - x) **2 )  + 100 * mean( np.diff(x) **2 )
#
##...............................................................................
#def nesterov( x ):
#    """ Nesterov's nonsmooth Chebyshev-Rosenbrock function, Overton 2011 variant 2 """
#    x = np.asarray_chkfinite(x)
#    x0 = x[:-1]
#    x1 = x[1:]
#    return abs( 1 - x[0] ) / 4 \
#        + sum( abs( x1 - 2*abs(x0) + 1 ))
#
##...............................................................................
#def saddle( x ):
#    x = np.asarray_chkfinite(x) - 1
#    return np.mean( np.diff( x **2 )) \
#        + .5 * np.mean( x **4 )
#
#
#
#    # bounds from Hedar, used for starting random_in_box too --
#    # getbounds evals ["-dim", "dim"]
#ackley_bounds       = [-15, 30]
#dixonprice_bounds   = [-10, 10]
#griewank_bounds     = [-600, 600]
#levy_bounds         = [-10, 10]
#michalewicz_bounds  = [0, pi]
#perm_bounds         = ["-dim", "dim"]  # min at [1 2 .. n]
#powell_bounds       = [-4, 5]  # min at tile [3 -1 0 1]
#powersum_bounds     = [0, "dim"]  # 4d min at [1 2 3 4]
#rastrigin_bounds    = [-5.12, 5.12]
#rosenbrock_bounds   = [-2.4, 2.4]  # wikipedia
#schwefel_bounds     = [-500, 500]
#sphere_bounds       = [-5.12, 5.12]
#sum2_bounds         = [-10, 10]
#trid_bounds         = ["-dim**2", "dim**2"]  # fmin -50 6d, -200 10d
#zakharov_bounds     = [-5, 10]
#
#ellipse_bounds      =  [-2, 2]
#logsumexp_bounds    = [-20, 20]  # ?
#nesterov_bounds     = [-2, 2]
#powellsincos_bounds = [ "-20*pi*dim", "20*pi*dim"]
#randomquad_bounds   = [-10000, 10000]
#saddle_bounds       = [-3, 3]



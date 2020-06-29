from arspy import ars

import numpy as np
from numpy import log, exp
from scipy.special import gamma as Gamma
from scipy.stats import gennorm


#So in our notations form_parameter=beta:=gamma. Also we fix scale_parameter = Gamma(beta).
#Generalized normal distribution is truncated into the interval [a,b].

#phi = lambda x, gamma=2: (gamma/(2*Gamma(1/gamma)))*exp(-np.abs(x)**gamma)
phi = lambda x, gamma: gennorm.pdf(x,gamma)
log_phi = lambda x, gamma: gennorm.logpdf(x,gamma)
Phi = lambda x, gamma: gennorm.cdf(x,gamma)
#TGN_pdf = lambda x, gamma,alpha,a,b:  1/(Phi((b-alpha)/(Gamma(gamma)),gamma)-Phi((a-alpha)/(Gamma(gamma)),gamma)) * (1/Gamma(gamma)) * phi((x-alpha)/Gamma(gamma),gamma) * int(a<=x<=b)
#log_TGN_pdf = lambda x, gamma,alpha,a,b: log(TGN_pdf(x, gamma,alpha,a,b))

#shape=(Gamma(gamma)*abs(b-a)/10)
#log_TGN_pdf = lambda x, gamma,alpha,a,b: log_phi((x-alpha)/Gamma(gamma),gamma) - log(Gamma(gamma)*(Phi((b-alpha)/(Gamma(gamma)),gamma)-Phi((a-alpha)/(Gamma(gamma)),gamma))) #numerically more stable!
log_TGN_pdf = lambda x, gamma,alpha,a,b: log_phi((x-alpha)/(Gamma(gamma)*abs(b-a)/10),gamma) - log((Gamma(gamma)*abs(b-a)/10)*(Phi((b-alpha)/((Gamma(gamma)*abs(b-a)/10)),gamma)-Phi((a-alpha)/((Gamma(gamma)*abs(b-a)/10)),gamma))) #numerically more stable!

def TGN_sample(size,gamma,alpha,x_min,x_max):
    domain = (x_min, x_max)
    return ars.adaptive_rejection_sampling(logpdf=lambda x: log_TGN_pdf(x,gamma,alpha,x_min,x_max), a=x_min, b=x_max, domain=domain, n_samples=size)


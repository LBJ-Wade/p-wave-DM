import numpy as np

def gaussian_normal(x):
    return np.exp(-0.5*(x)**2)/np.sqrt(2.0*np.pi)
    
def get_integral(x,g):
    print x.shape
    print g.shape
    return sum(np.diff(x)*0.5*(g[0:len(g)-1]+g[1:len(g)]))

def sigma_given_p(p):
    x = np.linspace(-200, 200, 50000)
    g = 1.0/np.sqrt(2*np.pi)*np.exp(-(x**2)/2.)
    c = np.cumsum(g)/sum(g)
    value = x[np.argmin(np.abs(c-(1.0-p)))]
    return value
    
def gamma(x):
	if x%1 == 0:
		return factorial(x-1)
	if x%1 == 0.5:
		return np.sqrt(np.pi)*factorial(2*(x-0.5))/(4**(x-0.5)*factorial((x-0.5)))
def chi_square_pdf(k,x):
    return 1.0/(2**(k/2)*gamma(k/2))*x**(k/2-1)*np.exp(-0.5*x)
def chi_square_cdf(k,x):
    return gammainc(k/2,x/2)
    
def chi_square_quantile(k,f):
    #Essentially do a numerical integral, until the value is greater than f
    integral_fraction = 0.0
    x = 0.0
    dx = 0.01
    while chi_square_cdf(k,x)<f:
        x += dx
    return x
    
trials = 318

local_sigma = 4.0

xspace = np.linspace(-10.0, local_sigma, 10000)
global_sig = sigma_given_p(1.0-get_integral(xspace, gaussian_normal(xspace))**trials)

print "Global significance = " + str(global_sig)
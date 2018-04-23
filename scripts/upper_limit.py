print("Loading required packages...")
#Math packages, NumPy
import math
import numpy as np
from BinnedAnalysis import *

#pfrom scipy.special import gammainc, erf, gamma
from math import sin, cos, asin, acos, radians
from scipy.misc import factorial
import scipy
from scipy.optimize import curve_fit
from operator import add
import pickle

#Fermi Science Tools
#from SummedLikelihood import *

#Matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from matplotlib.pyplot import rc
from matplotlib import rcParams
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Wedge, Circle

from astropy.io import fits
from astropy.wcs import WCS
from astropy.utils.data import get_pkg_data_filename
from astropy.coordinates import SkyCoord

print("Done!")

gc_l = 359.94425518526566
gc_b = -0.04633599860905694
gc_ra = 266.417
gc_dec = -29.0079

def setup_plot_env():
    #Set up figure
    #Plotting parameters
    fig_width = 8   # width in inches
    fig_height = 8  # height in inches
    fig_size =  [fig_width, fig_height]
    rcParams['font.family'] = 'serif'
    rcParams['font.weight'] = 'bold'
    rcParams['axes.labelsize'] = 24
    rcParams['font.size'] = 26
    rcParams['axes.titlesize'] =16
    rcParams['legend.fontsize'] = 16
    rcParams['xtick.labelsize'] =28
    rcParams['ytick.labelsize'] =28
    rcParams['figure.figsize'] = fig_size
    rcParams['xtick.major.size'] = 14
    rcParams['ytick.major.size'] = 14
    rcParams['xtick.minor.size'] = 4
    rcParams['ytick.minor.size'] = 4
    rcParams['xtick.major.pad'] = 8
    rcParams['ytick.major.pad'] = 8

    rcParams['figure.subplot.left'] = 0.16
    rcParams['figure.subplot.right'] = 0.92
    rcParams['figure.subplot.top'] = 0.90
    rcParams['figure.subplot.bottom'] = 0.12
    rcParams['text.usetex'] = True
    rc('text.latex', preamble=r'\usepackage{amsmath}')
setup_plot_env()

#A derived class from the Science Tools BinnedAnalysis class
#Adds various methods which make life easier
class AnalyticAnalysis(BinnedAnalysis):

    #Methods to loop over the sources & freeze them all
    def freeze_all_sources(self):
        for k in range(len(self.params())):
            self.freeze(k)

    def free_all_sources(self):
        for k in range(len(self.params())):
            if self.params()[k].parameter.getName()!='Scale' and self.params()[k].parameter.getName()!='Eb':
                self.thaw(k)

    def edit_parameter(self, source_name, parameter_name, new_value):
        self[source_name]['Spectrum'][parameter_name] = new_value
        #self.fit(verbosity=0)

    #The fit & optimize methods of BinnedAnalysis already return the minus loglikelihood, but it's unclear exactly how it is calculated
    #Here it's just calculated from the definition
    def loglikelihood(self):
        #The sourcemap file is needed to calculate the model.
        sourcemap = self.binnedData.srcMaps
        f = fits.open(sourcemap)

        #Given the results of the fit, calculate the model
        model_data = np.zeros(f[0].shape)
        for source in self.sourceNames():
            the_index = f.index_of(source)
            model_data += self._srcCnts(source)[:, None, None]*f[the_index].data[:-1, :, :]/np.sum(np.sum(f[the_index].data, axis=2), axis=1)[:-1, None, None]
        actual_data = np.array(self.binnedData.countsMap.data()).reshape(f[0].shape)
        #Likelihood value is a product of Poisson factors
        likelihood = np.sum(np.sum(np.sum(np.log(model_data**actual_data*np.exp(-1.0*model_data)/factorial(actual_data)))))

        #Return minus log-likelihood- a function to be minimized
        return -1.0*likelihood

    #Calculate the covariance matrix between two sources with a simple difference method. Works on boundary values as well
    def calculateCovarianceMatrix(self, source_A, parameter_A, source_B, parameter_B):
        current_val_a = self[source_A]['Spectrum'][parameter_A]
        current_val_b = self[source_B]['Spectrum'][parameter_B]

        #First, calculate Hessian to a) confirm we are at a minimum and
        #b) covariance matrix is the inverse of the hessian
        num_parameters = 2
        hessian = np.zeros((num_parameters, num_parameters))

        dx_a = 100.0
        dx_b = np.abs(0.1*current_val_b)

        for a in range(num_parameters):
            for b in range(num_parameters):
                print(a, b)
                if a == b:
                    if a == 0:
                        print(self[source_A]['Spectrum'][parameter_A])
                        self.edit_parameter(source_A, parameter_A, current_val_a+2*dx_a)
                        print(self[source_A]['Spectrum'][parameter_A])
                        yiplus1 = self.loglikelihood()

                        self.edit_parameter(source_A, parameter_A, current_val_a+dx_a)
                        yi = self.loglikelihood()

                        self.edit_parameter(source_A, parameter_A, current_val_a)
                        yiminus1 = self.loglikelihood()
                        print(y1plus1, yi, yiminus1)
                        hessian[a,b] = (yiplus1-2.0*yi+yiminus1)/(dx_a**2)
                    if a == 1:
                        self.edit_parameter(source_B, parameter_B, current_val_b+dx_b)
                        yiplus1 = self.loglikelihood()

                        self.edit_parameter(source_B, parameter_B, current_val_b)
                        yi = self.loglikelihood()

                        self.edit_parameter(source_B, parameter_B, current_val_b-dx_b)
                        yiminus1 = self.loglikelihood()

                        hessian[a,b] = (yiplus1-2.0*yi+yiminus1)/(dx_b**2)

                else:
                    z = np.zeros((2,2))
                    self.edit_parameter(source_A, parameter_A, current_val_a+2*dx_a)
                    self.edit_parameter(source_B, parameter_B, current_val_b+dx_b)
                    z[0,0] = self.loglikelihood()

                    self.edit_parameter(source_A, parameter_A, current_val_a+2*dx_a)
                    self.edit_parameter(source_B, parameter_B, current_val_b-dx_b)
                    z[0,1] = self.loglikelihood()

                    self.edit_parameter(source_A, parameter_A, current_val_a)
                    self.edit_parameter(source_B, parameter_B, current_val_b+dx_b)
                    z[1,0] = self.loglikelihood()

                    self.edit_parameter(source_A, parameter_A, current_val_a)
                    self.edit_parameter(source_B, parameter_B, current_val_b-dx_b)
                    z[1,1] = self.loglikelihood()
                    hessian[a,b] = (z[1,1]-z[0,1]-z[1,0]+z[0,0])/(4.0*dx_a*dx_b)

        if np.sign(np.linalg.det(hessian)) == -1.0:
            print("WARNING: Determinant of Hessian matrix is negative- you are probably not at an extremum of the function you are trying to optimize")
            return 0
        print("hessian = " + str(hessian))

        #Covariance matrix is the inverse of the hessian
        covariance = np.linalg.inv(hessian)
        self.covariance = covariance

    def getCovarianceMatrix(self):
        return self.covariance

    def calculateCorrelationMatrix(self, covariance):
        correlation = np.zeros((covariance.shape))
        for a in covariance.shape[0]:
            for b in covariance.shape[1]:
                correlation[a,b] = covariance[a][b]/np.sqrt(float(covariance[a][a])*float(covariance[b][b]))
        self.correlation = correlation

    def getCorrelationMatrix(self):
        return self.correlation

    def getSpectrum(self):
        spectrum = np.zeros((num_ebins-1))
        for source in self.sourceNames():
            spectrum += self._srcCnts(source)
        return spectrum

    #Make a plot of the data/model/residuals
    def roiPlot(self):
        sourcemap = self.binnedData.srcMaps
        f = fits.open(sourcemap)

        image_data = fits.getdata('6gev_image.fits')
        filename = get_pkg_data_filename('6gev_image.fits')
        hdu = fits.open(filename)[0]
        wcs = WCS(hdu.header)
        #Given the results of the fit, calculate the model
        model_data = np.zeros(f[0].shape)
        for source in self.sourceNames():
            the_index = f.index_of(source)
            model_data += self._srcCnts(source)[:, None, None]*f[the_index].data[:-1, :, :]/np.sum(np.sum(f[the_index].data, axis=2), axis=1)[:-1, None, None]
        actual_data = np.array(self.binnedData.countsMap.data()).reshape(f[0].shape)

        fig = plt.figure(figsize=[14,6])

        ax = fig.add_subplot(131, projection=wcs.wcs)
        ax=plt.gca()

        c = Wedge((gc_l, gc_b), 1.0, theta1=0.0, theta2=360.0, width=14.0, edgecolor='black', facecolor='#474747', transform=ax.get_transform('galactic'))
        ax.add_patch(c)
        mappable=plt.imshow(np.sum(actual_data, axis=0),cmap='inferno',origin='lower',norm=colors.PowerNorm(gamma=0.6),vmin=0, vmax=65, interpolation='gaussian')#
        plt.xlabel('Galactic Longitude')
        plt.ylabel('Galactic Latitude')
        plt.title('Data')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cb = plt.colorbar(mappable, cax=cax, label='Counts per pixel')

        ax2=fig.add_subplot(132, projection=wcs.wcs)
        ax2 = plt.gca()
        c2 = Wedge((gc_l, gc_b), 1.0, theta1=0.0, theta2=360.0, width=14.0, edgecolor='black', facecolor='#474747', transform=ax2.get_transform('galactic'))
        ax2.add_patch(c2)
        mappable2 = plt.imshow(np.sum(model_data, axis=0), cmap='inferno',origin='lower',norm=colors.PowerNorm(gamma=0.6),vmin=0, vmax=65, interpolation='gaussian')
        plt.xlabel('Galactic Longitude')
        plt.ylabel('Galactic Latitude')
        plt.title('Model')
        divider2 = make_axes_locatable(ax2)
        cax2 = divider2.append_axes("right", size="5%", pad=0.05)
        cb2 = plt.colorbar(mappable2, cax=cax2, label='Counts per pixel')

        ax3=fig.add_subplot(133, projection=wcs.wcs)
        ax3 = plt.gca()
        c3 = Wedge((gc_l, gc_b), 1.0, theta1=0.0, theta2=360.0, width=14.0, edgecolor='black', facecolor='#474747', transform=ax3.get_transform('galactic'))
        ax3.add_patch(c3)
        mappable3 = plt.imshow(np.sum(actual_data, axis=0)-np.sum(model_data, axis=0), cmap='seismic',origin='lower', vmin=-20, vmax=20, interpolation='gaussian')#
        plt.xlabel('Galactic Longitude')
        plt.ylabel('Galactic Latitude')
        plt.title('Residuals')
        divider3 = make_axes_locatable(ax3)
        cax3 = divider3.append_axes("right", size="5%", pad=0.05)
        cb3 = plt.colorbar(mappable3, cax=cax3, label='Counts per pixel')
        fig.tight_layout()

        plt.show()

    def likelihoodMap(self, source_A, parameter_A, source_B, parameter_B, res = 10, verbose=False, generate_plot=False):
        print("Making likelihood map...")
        like_mat = np.zeros((res,res))
        l = 0
        k = 0

        #Find a good range of parameters from the Hessian
        cutoff_delta_loglike = 10.0
        xspace = np.linspace(max(0,current_val_a-np.sqrt(2*cutoff_delta_loglike/hessian[0,0])), current_val_a+np.sqrt(2*cutoff_delta_loglike/hessian[0,0]), res)
        yspace = np.linspace(current_val_b-np.sqrt(2*cutoff_delta_loglike/hessian[1,1]), current_val_b+np.sqrt(2*cutoff_delta_loglike/hessian[1,1]), res)
        for xval in xspace:
            l = 0
            for yval in yspace:
                self.edit_parameter(source_A, parameter_A, xval)
                self.edit_parameter(source_B, parameter_B, yval)

                like_mat[k, l] = self.loglikelihood()
                if verbose:
                    print(source_B + " " + parameter_B + " = " + str(self[source_B]['Spectrum'][parameter_B]))
                    print(source_A + " " + parameter_A + " = " + str(self[source_A]['Spectrum'][parameter_A]))
                l += 1
            k += 1

        fig = plt.figure(figsize=[10,10])
        ax = fig.add_subplot(111)
        mappable = plt.imshow(like_mat, origin='lower', extent=[min(yspace), max(yspace), min(xspace), max(xspace)], aspect='auto')
        plt.colorbar(mappable, label='LogLikelihood')
        plt.xlabel(source_B + " " + parameter_B)
        plt.ylabel(source_A + " " + parameter_A)

        self.edit_parameter(source_A, parameter_A, current_val_a)
        self.edit_parameter(source_B, parameter_B, current_val_b)

        fig.tight_layout()
        plt.show()


#Strictly finds the Poisson upper limit (problematic for large counts)
def upper_limit(N,conf,b):
    #Calculates upper limit on signal s, given background b, number of events N, and confidence interval conf (95% = 0.05)
    #Decides whether the value is really big, in which case do everything with ints. Otherwise, use floats
    #First, calculate denominator
    denom=0.
    for m in range(0,N+1):
            denom+=b**m/math.factorial(m)
    s = 0.
    numer=denom
    while math.exp(-1.0*s)*numer/denom>conf:
        #Calculate numerator
        numer=0.0
        for m in range(0,N+1):
            numer+=(s+b)**m/math.factorial(m)
        s+= 0.01
    print("Upper limit is " + str(s))
    return s

def factorial2(x):
	result = 1.
	while x>0:
		result *= x
		x -= 1.
	return result
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
def upper_limit_pdg(N,alpha,b):
    dof = 2*(N+1)
    p = 1-alpha*(1-(chi_square_cdf(dof, 2*b)))
    sup = 0.5*chi_square_quantile(dof, p)-b
    return sup

def chisquared(data,theory):
    thesum = 0.0
    for i in range(len(data)):
        if theory[i]>0.0:
            thesum += (theory[i]-data[i])**2/(float(theory[i]))
    return thesum
def loglikelihood(counts,model_counts,box):
    f0 = 0.0
    f1 = 0.0
    for i in range(len(counts)):
        #m = number of counts
        m = counts[i]
        #null hypothesis counts:
        mu0 = model_counts[i]
        #background+signal counts
        mu1 = model_counts[i]+box[i]
        #Do we need stirling's approximation?
        if m>20:
            f0 += m-mu0+m*np.log(mu0/m)-0.5*np.log(m)-np.log(np.sqrt(2*np.pi))
            f1 += m-mu1+m*np.log(mu1/m)-0.5*np.log(m)-np.log(np.sqrt(2*np.pi))
        else:
            f0 += np.log((mu0**m)*np.exp(-1.0*mu0)/factorial(m))
            f1 += np.log((mu1**m)*np.exp(-1.0*mu1)/factorial(m))
    return 2*(f0-f1)

def sigma_given_p(p):
    x = np.linspace(-200, 200, 50000)
    g = 1.0/np.sqrt(2*np.pi)*np.exp(-(x**2)/2.)
    c = np.cumsum(g)/sum(g)
    value = x[np.argmin(np.abs(c-(1.0-p)))]
    return value
def pvalue_given_chi2(x, k):
    y = np.arange(0., 1000.0, 0.1)
    g = (y**(k/2.-1.0)*np.exp(-0.5*y))/(2.**(k/2.0)*gamma(k/2.))
    initial_pos = np.argmin(np.abs(y-x))
    return get_integral(y[initial_pos:], g[initial_pos:])
#Stupid function bc apparently numpy can't do this natively??
def multiply_multidimensional_array(vec,cube):
    result = np.zeros((cube.shape))
    for i in range(len(vec)):
        result[i,:,:] = vec[i]*cube[i,:,:]
    return result
#Energy resolution (E Disp class 1- pretty close to the total)
def e_res(E):
    energy = np.array([31.718504,54.31288,95.86265,171.82704,307.93054,535.1251,944.0604,1716.8848,3074.5315,5339.1763,9559.056,17111.936,29704.664,53986.227,93706.14,167702.28,300152.7,520977.8,931974.8,1690808.0,2974747.2])
    res = np.array([0.2942218,0.25078142,0.21741302,0.1813951,0.15067752,0.12313887,0.10302135,0.090325624,0.08080946,0.07341215,0.07025641,0.070810914,0.07454435,0.080399856,0.08678346,0.094758466,0.10061333,0.10752697,0.25969607,0.18390165,0.24169934])
    closest = np.argmin(np.abs(E-energy))
    if E-energy[closest]>0.:
        frac = (E-energy[closest])/(energy[closest+1]-energy[closest])
        return res[closest]+frac*(res[closest+1]-res[closest])
    else:
        frac = (E-energy[closest-1])/(energy[closest]-energy[closest-1])
        return res[closest-1]+frac*(res[closest]-res[closest-1])
def psf(E):
    energy = np.array([9.91152,17.36871,31.150045,54.59528,96.42895,171.62605,303.14316,539.58026,967.85913,1709.5619,3066.256,5374.1895,9712.058,17151.041,29366.348,52649.074,92947.98,167911.25,298723.0,527422.3,952855.0,1682382.6,2993103.8])
    psf = np.array([22.122343,17.216175,11.960119,8.108732,5.279108,3.5216076,2.2375877,1.3988715,0.8535155,0.53358656,0.347393,0.23173566,0.17039458,0.12837319,0.112826064,0.10581638,0.10334797,0.10426899,0.10101496,0.09097172,0.08671612,0.07683781,0.073241934])
    closest = np.argmin(np.abs(E-energy))
    if E-energy[closest]>0.:
        frac = (E-energy[closest])/(energy[closest+1]-energy[closest])
        return psf[closest]+frac*(psf[closest+1]-psf[closest])
    else:
        frac = (E-energy[closest-1])/(energy[closest]-energy[closest-1])
        return psf[closest-1]+frac*(psf[closest]-psf[closest-1])
#Just a gaussian function, representing the energy dispersion of the detector
def blur(x,offset,sigma):
    return np.exp(-1.0*(x-offset)**2/(2*sigma**2))/np.sqrt(2*np.pi*sigma**2)

def make_random(x,g):
    cdf = np.cumsum(g)/np.sum(g)
    return x[np.argmin(np.abs(cdf-np.random.rand(1)[0]))]
def get_integral(x,g):
    if len(x) != len(g):
        print("Integral must be performed with equal-sized arrays!")
        print("Length of x is " + str(len(x)) + " Length of g is " + str(len(g)))
    return sum(np.diff(x)*0.5*(g[0:len(g)-1]+g[1:len(g)]))

def update_box_spectrum(energy, zeta):
    box_width = energy*2.0*np.sqrt(1.0-zeta)/(1+np.sqrt(1.0-zeta))
    box_beginning = energy-box_width
    #Edit box_spectrum.dat
    spectrum_file = open('box_spectrum.dat','w')
    #What's the normalization constant here?
    #Total flux is E_edge*function value
    #Function_value = total_flux/E_edge
    x_fine_grid = np.linspace(0.0, 800000, 10000)
    n_leading_zeros = int(np.argmin(np.abs(x_fine_grid-box_beginning)))
    n_trailing_zeros = 10000-int(np.argmin(np.abs(x_fine_grid-energy)))
    n_box_width = int(10000-n_leading_zeros-n_trailing_zeros)
    pure_box = np.concatenate([np.zeros((n_leading_zeros)),np.concatenate([1.0+np.zeros((n_box_width)),np.zeros((n_trailing_zeros))])])

    #Sigma here is the absolute energy resolution as a function of energy
    sigma = e_res(energy)*energy*10000./800000.0

    dispersion = blur(np.linspace(0,6*sigma,6*sigma),3*sigma,sigma)
    convolved_pure_box = np.convolve(pure_box, dispersion,'same')
    integrated_box_flux = get_integral(x_fine_grid, convolved_pure_box)
    for i in range(len(x_fine_grid)):
        spectrum_file.write(str(x_fine_grid[i])+" " + str(max(convolved_pure_box[i]/integrated_box_flux, 10.**-35))+"\n")
    spectrum_file.close()

    return box_width, integrated_box_flux

def quadratic(x, a, b, c):
    return a*x**2+b*x+c

num_ebins = 51 #1 more than the number of bins due to the fencepost problem
energies = 10**np.linspace(np.log10(6000),np.log10(800000),num_ebins)
ebin_widths = np.diff(energies)

def likelihood_upper_limit3(zeta, sourcemap, poisson=False):
    scale_factor = 1e-15
    crit_chi2 = 2.71 #For 95% confidence one-sided upper limit with 1 degree of freedom

    #Arrays to store results in
    box_flux = np.zeros((num_ebins-1, 3))
    correlations = np.zeros((num_ebins, 2))
    #sourcemap = '6gev_srcmap_03.fits'

    #Instantiate the analysis objects
    print("Loading data...")
    obs_complete = BinnedObs(srcMaps=sourcemap, expCube='6gev_ltcube.fits', binnedExpMap='6gev_exposure.fits', irfs='P8R2_SOURCE_V6')
    print("obs loaded")
    like = AnalyticAnalysis(obs_complete, 'xmlmodel.xml', optimizer='MINUIT')

    #For MC study i.e. making Brazil plot bands
    if poisson:
        print("Calculating fluctuations...")
        #Poisson fluctuations of the data
        f = fits.open(sourcemap)
        #Given the results of the fit, calculate the model
        model_data = np.zeros(f[0].shape)
        for source in like.sourceNames():
            the_index = f.index_of(source)
            model_data += like._srcCnts(source)[:, None, None]*f[the_index].data[:-1, :, :]/np.sum(np.sum(f[the_index].data, axis=2), axis=1)[:-1, None, None]

        #Introduce Poisson fluctuations on top of the model
        poisson_data = np.random.poisson(model_data)

        #Write to a new sourcemap
        f[0].data = poisson_data
        f.writeto('box_srcmap_poisson.fits')
        f.close()
        obs_poisson = BinnedObs(srcMaps='box_srcmap_poisson.fits', expCube='6gev_ltcube.fits', binnedExpMap='6gev_exposure.fits', irfs='P8R2_SOURCE_V6')

    #Loop through upper edge of box
    for index in range(29,48):
        print("Evaluating box with upper edge " + str(energies[index]) + " MeV in bin " + str(index))
        #Update the box spectrum
        box_width, integrated_box_flux = update_box_spectrum(energies[index], zeta)
    	like.free_all_sources()
    	#like.freeze(like.par_index('Box_Component','Normalization'))
    	likeobj = pyLike.Minuit(like.logLike)
    	loglike = like.fit(verbosity=0, optObject=likeobj, covar=False)

    	#print('loglike = ' + str(loglike))
    	#while loglike>15830.0:
    	#    loglike = like.fit(verbosity=0, optObject=likeobj, covar=False)
    	#    print('loglike = '+str(loglike))
    	#like.freeze_all_sources()
    	#print(dir(like))
    	#myDict = {}
            #for source in like.sourceNames():
    	#    print(like._srcCnts(source))
    	#    myDict[source] = like._srcCnts(source)
    	#file = open('fitResults001.pk1','wb')
    	#pickle.dump(myDict,file)
    	#file.close()
    	#input('wait for key')

    	if poisson:
                like = AnalyticAnalysis(obs_poisson, 'xmlmodel.xml', optimizer='NewMinuit')
                print("Fitting...")
                likeobj = pyLike.Minuit(like.logLike)
                like.free_all_sources()
                like.tol=1e-9
                loglike = like.fit(verbosity=0,optObject=likeobj, covar=False)
                print("Return code: " + str(likeobj.getRetCode()))
                print("loglike = " + str(loglike))
                while loglike> 18000.0:
                    loglike = like.fit(verbosity=0,optObject=likeobj, covar=False)
                    print("Return code: " + str(likeobj.getRetCode()))
                    print("loglike = " + str(loglike))
                like.freeze_all_sources()

        #Output spectrum info
        box_spectrum = self._srcCnts('Box_Component')
        window_spectrum = self.getSpectrum()
        complete_spectrum = self.nobs

        like.thaw(like.par_index('Disk Component','Prefactor'))
        print("Scanning likelihood...")
        #Scan the likelihood profile to find best-fit value and upper limit
        minx = 0.0
        maxx = 10**-9/scale_factor
        x_range, l_range = like.scan('Box_Component', 'Normalization', minx, maxx, 200)
        #maxx = x_range[max(1, np.argmin(np.abs(l_range-(crit_chi2+min(l_range)))))]*2.0

        #Results
        box_flux[index, 0] = x_range[np.argmin(np.abs(l_range-(crit_chi2+min(l_range))))]*scale_factor #Flux upper limit
        box_flux[index, 1] = x_range[np.argmin(l_range)]*scale_factor #Best fit flux
        box_flux[index, 2] = l_range[np.argmin(l_range)]-l_range[0] #Delta log like for best-fit flux
        print("Best-fit box = " + str(box_flux[index,1]))
        if box_flux[index,1]>0.0:
            print("Significance = " + str(sigma_given_p(pvalue_given_chi2(-2.0*box_flux[index,2], 1))) + " sigma")
        print("Box upper limit = " + str(box_flux[index, 0]))

        print("Calculating correlation matrix...")
        #Correlations between the sources
        #like.calculateCovarianceMatrix('Box_Component', 'Normalization', 'Disk Component', 'Prefactor')
        #like.calculateCorrelationMatrix()
        #correlations[index, 0] = like.correlation[0,1]

        #like.calculateCovarianceMatrix('Box_Component', 'Normalization', 'Disk Component', 'Index')
        #like.calculateCorrelationMatrix()
        #correlations[index, 1] = like.correlation[0,1]

        print("Correlation with GC Prefactor: " + str(correlations[index,0]))
        print("Correlation with GC Index: " + str(correlations[index,0]))


    return box_flux, correlations



def main():
    #[z99_ul, z99_corr] = likelihood_upper_limit3(0.9999, sourcemap, poisson=True)
    #file = open('/nfs/farm/g/glast/u/johnsarc/p-wave_DM/6gev/z99.pk1','wb')
    #pickle.dump([z99_ul, z99_corr],file)
    #file.close()

    #sourcemap = '6gev_srcmap_03.fits'
    #[z44_ul, z44_corr] = likelihood_upper_limit3(0.44, sourcemap)
    #file = open('/nfs/farm/g/glast/u/johnsarc/p-wave_DM/6gev/z44.pk1','wb')
    #pickle.dump([z44_ul, z44_corr],file)
    #file.close()

    #sourcemap = '6gev_srcmap_001.fits'
    sourcemap = 'box_srcmap_artificial_box.fits'
    [z44_artificial_ul, z44_artificial_corr] = likelihood_upper_limit3(0.44, sourcemap, poisson=False)
    #file = open('/nfs/farm/g/glast/u/johnsarc/p-wave_DM/6gev/z44_poisson.pk1','wb')
    #pickle.dump([z44_artificial_ul, z44_artificial_corr],file)
    #file.close()
    #raw_input('wait for key')
    #plotter([z0, z44, z85, z99])
    input('wait for key')




    brazil_dict = consolidate_brazil_lines('brazil_wide_box.pk1')
    file = open('z44.pk1','rb')
    [z44_ul, z44_corr_blank] = pickle.load(file)
    file.close()
    #Did this twice because you want to use NewMinuit to set limits, and MINUIT for correlations
    file = open('z44_corr.pk1','rb')
    [z44_ul_blank, z44_corr] = pickle.load(file)
    file.close()
    #correlationPlot(z44_corr, z99_corr)
    #make_ul_plot(z44_ul[:,0],brazil_dict,'UL','brazil_wide_box.pdf')
    #make_ul_plot(z44_ul,brazil_dict,'SIG', 'significance_wide_box.pdf')
    print("z44 significances = " + str(z44_ul[:,2]))
    print("Most signficant result = " + str(sigma_given_p(pvalue_given_chi2(-2.0*min(z44_ul[:,2]),1)))+ " sigma")
    print("at position " + str(energies[np.argmin(z44_ul[:,2])]) + " MeV")

    brazil_dict = consolidate_brazil_lines('brazil_narrow_box.pk1')
    file = open('z99.pk1','rb')
    [z99_ul, z99_corr_blank] = pickle.load(file)
    file.close()
    file = open('z99_corr.pk1','rb')
    [z99_ul_blank, z99_corr] = pickle.load(file)
    file.close()
    print("z99 significances = " + str(z99_ul[:,2]))
    print("Most signficant result = " + str(sigma_given_p(pvalue_given_chi2(-2.0*min(z99_ul[:,2]),1)))+ " sigma")
    print("at position " + str(energies[np.argmin(z99_ul[:,2])]) + " MeV")
    #make_ul_plot(z99_ul[:,0],brazil_dict,'UL','brazil_narrow_box.pdf')
    #make_ul_plot(z99,brazil_dict,'SIG', 'significance_narrow_box.pdf')
    correlationPlot(z44_corr, z99_corr)
    file = open('z_artificial.pk1','rb')
    [z44_artificial_ul, z44_artificial_corr] = pickle.load(file)
    file.close()
    brazil_dict = consolidate_brazil_lines('brazil_artificial_box.pk1')
    print("Most signficant result = " + str(sigma_given_p(pvalue_given_chi2(-2.0*min(z44_artificial_ul[:,2]),1)))+ " sigma")
    print("at position " + str(energies[np.argmin(z44_artificial_ul[:,2])]) + " MeV")

    #make_ul_plot(z44_artificial_ul[:,0], brazil_dict,'UL','brazil_artificial_box.pdf')
    #make_ul_plot(z_artificial_box, brazil_dict,'SIG', 'significance_artificial_box.pdf')

if __name__ == '__main__':
    main()
